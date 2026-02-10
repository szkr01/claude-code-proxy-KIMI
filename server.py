"""
Claude Code Proxy with Kimi OAuth Integration

Anthropic API形式でKimi APIを使用できるプロキシサーバー
OAuth Device Authorization Grantによる自動認証をサポート
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

import httpx
import litellm
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from token_manager import TokenManager, OAuthToken
from kimi_auth import KimiAuth, KimiAuthError, KimiAuthUnauthorized

# 環境変数を読み込み
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Uvicornログを抑制
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# 定数
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8082
DEFAULT_AUTO_LOGIN = "true"
DEFAULT_AUTO_OPEN_BROWSER = "true"

# グローバル設定（コマンドライン引数から設定）
CLI_LOG_CONVERSATIONS = False
CLI_LOG_DIR = Path("logs/conversations")

# Kimiモデルリスト
KIMI_MODELS = [
    "kimi-k2-turbo-preview",
    "kimi-k2.5",
    "kimi-for-coding",
    "kimi-k2-turbo",
]

# OpenAIモデルリスト（フォールバック用）
OPENAI_MODELS = [
    "o3-mini", "o1", "o1-mini", "o1-pro",
    "gpt-4.5-preview", "gpt-4o", "gpt-4o-mini",
    "gpt-4.1", "gpt-4.1-mini",
]

# Geminiモデルリスト（フォールバック用）
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


# ============================================================================
# 会話ログ機能
# ============================================================================

def save_conversation_log(
    request_data: dict[str, Any],
    response_data: dict[str, Any],
    log_dir: Path,
    is_stream: bool = False,
) -> None:
    """会話ログを保存"""
    try:
        # ログディレクトリを作成
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # ログファイル名（タイムスタンプ + UUID）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_id = uuid.uuid4().hex[:8]
        log_file = log_dir / f"conversation_{timestamp}_{log_id}.json"
        
        # ログデータを構築
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data,
            "is_stream": is_stream,
        }
        
        # ファイルに保存
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Conversation log saved: {log_file}")
    
    except Exception as e:
        logger.warning(f"Failed to save conversation log: {e}")


# ============================================================================
# Pydantic Models
# ============================================================================

class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int = 32000  # Kimiのデフォルトに合わせる（kimi-cliと同じ）
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None
    
    @field_validator("model")
    def validate_model_field(cls, v: str, info) -> str:
        """モデル名を検証・変換"""
        original_model = v
        new_model = v
        
        # プロバイダー接頭辞を削除
        clean_v = v
        if clean_v.startswith("anthropic/"):
            clean_v = clean_v[10:]
        elif clean_v.startswith("openai/"):
            clean_v = clean_v[7:]
        elif clean_v.startswith("gemini/"):
            clean_v = clean_v[7:]
        elif clean_v.startswith("kimi/"):
            clean_v = clean_v[5:]
        
        # マッピングロジック
        mapped = False
        
        # 環境変数から設定を取得
        preferred = os.getenv("PREFERRED_PROVIDER", "kimi").lower()
        big_model = os.getenv("BIG_MODEL", "kimi-k2-turbo-preview")
        small_model = os.getenv("SMALL_MODEL", "kimi-k2-turbo-preview")
        
        # Kimiが優先プロバイダーの場合
        if preferred == "kimi":
            # ClaudeモデルをKimiモデルにマッピング
            if "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
                new_model = f"kimi/{big_model}"
                mapped = True
            elif "haiku" in clean_v.lower():
                new_model = f"kimi/{small_model}"
                mapped = True
            elif clean_v in KIMI_MODELS:
                new_model = f"kimi/{clean_v}"
                mapped = True
            else:
                # デフォルトでKimiモデル
                new_model = f"kimi/{big_model}"
                mapped = True
        
        # OpenAIが優先の場合
        elif preferred == "openai":
            if "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
                new_model = f"openai/{big_model}"
                mapped = True
            elif "haiku" in clean_v.lower():
                new_model = f"openai/{small_model}"
                mapped = True
            elif clean_v in OPENAI_MODELS:
                new_model = f"openai/{clean_v}"
                mapped = True
        
        # Googleが優先の場合
        elif preferred == "google":
            if "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
                new_model = f"gemini/{big_model}"
                mapped = True
            elif "haiku" in clean_v.lower():
                new_model = f"gemini/{small_model}"
                mapped = True
            elif clean_v in GEMINI_MODELS:
                new_model = f"gemini/{clean_v}"
                mapped = True
        
        # Anthropic直接使用
        elif preferred == "anthropic":
            new_model = f"anthropic/{clean_v}"
            mapped = True
        
        if mapped:
            logger.info(f"MODEL MAPPING: '{original_model}' -> '{new_model}'")
        
        # 元のモデル名を保存
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model
        
        return new_model


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# ============================================================================
# Kimi LiteLLM 設定
# ============================================================================

class KimiLiteLLMConfig:
    """LiteLLM経由でKimi APIを呼び出すための設定"""
    
    def __init__(
        self,
        kimi_auth: KimiAuth,
        log_conversations: bool = False,
        log_dir: Path = Path("logs/conversations"),
    ):
        self.kimi_auth = kimi_auth
        self.log_conversations = log_conversations
        self.log_dir = log_dir
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTPクライアントを取得"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def completion(self, **kwargs) -> Any:
        """
        Kimi API呼び出し（直接OpenAI互換API使用）
        
        LiteLLMをバイパスして直接Kimi APIを呼び出す
        """
        model = kwargs.get("model", "")
        
        if model.startswith("kimi/"):
            # Kimiモデルの場合、直接OpenAI互換APIを使用
            actual_model = model[5:]  # "kimi/"を削除
            
            # トークンを取得（自動更新）
            access_token = await self.kimi_auth.ensure_fresh_token()
            
            # OpenAI互換のリクエストを構築
            # TODO: 一時的にストリーミングを無効化してtool calls動作確認
            request_data = {
                "model": actual_model,
                "messages": kwargs.get("messages", []),
                "max_tokens": kwargs.get("max_tokens", 32000),
                "temperature": kwargs.get("temperature", 1.0),
                "stream": False,  # 強制的に非ストリーミングモードに設定
                # thinking機能を無効化
                "thinking": {
                    "type": "disabled"
                }
            }
            
            # オプショナルパラメータ
            if "tools" in kwargs and kwargs["tools"]:
                request_data["tools"] = kwargs["tools"]
                logger.info(f"Sending {len(kwargs['tools'])} tools to Kimi API")
            
            # tool_choiceは送信しない（Kimi APIがサポートしていない可能性）
            
            if "top_p" in kwargs:
                request_data["top_p"] = kwargs["top_p"]
            if "stop" in kwargs:
                request_data["stop"] = kwargs["stop"]
            # stream_options は非ストリーミングでは不要

            
            # デバッグログ
            logger.info(f"Direct Kimi API call: model={actual_model}, messages={len(request_data['messages'])}, tools={len(request_data.get('tools', []))}")
            
            # ヘッダー
            headers = self.kimi_auth._get_common_headers()
            headers["Authorization"] = f"Bearer {access_token}"
            headers["Content-Type"] = "application/json"
            
            # APIリクエスト
            client = await self._get_client()
            try:
                response = await client.post(
                    f"{self.kimi_auth.api_base_url}/chat/completions",
                    json=request_data,
                    headers=headers,
                )
                
                # デバッグ: レスポンスステータスとボディを確認
                logger.info(f"Kimi API response status: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                logger.debug(f"Response text (first 500 chars): {response.text[:500]}")
                
                response.raise_for_status()
                
                # JSONパース前にテキストを確認
                if not response.text:
                    raise ValueError("Empty response from Kimi API")
                    
                result = response.json()
                
                logger.info(f"Kimi API success: id={result.get('id')}, finish_reason={result.get('choices', [{}])[0].get('finish_reason')}")
                
                # tool_callsをチェック
                if result.get('choices') and result['choices'][0].get('message', {}).get('tool_calls'):
                    logger.info(f"Received {len(result['choices'][0]['message']['tool_calls'])} tool calls from Kimi")
                
                # 非ストリーミングの場合、SimpleNamespaceでラップ
                # 現在は強制的に非ストリーミングモード
                from types import SimpleNamespace
                
                # usage情報を構築
                usage_data = result.get("usage", {})
                usage = SimpleNamespace(**usage_data) if usage_data else None
                
                # choicesを構築
                choices = []
                for choice_data in result.get("choices", []):
                    message_data = choice_data.get("message", {})
                    
                    # tool_callsを変換
                    tool_calls = None
                    if "tool_calls" in message_data and message_data["tool_calls"]:
                        tool_calls = []
                        for tc in message_data["tool_calls"]:
                            function_data = tc.get("function", {})
                            function = SimpleNamespace(
                                name=function_data.get("name"),
                                arguments=function_data.get("arguments", "{}")
                            )
                            tool_call = SimpleNamespace(
                                id=tc.get("id"),
                                type=tc.get("type", "function"),
                                function=function
                            )
                            tool_calls.append(tool_call)
                    
                    # messageを構築
                    message = SimpleNamespace(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content"),
                        tool_calls=tool_calls
                    )
                    
                    choice = SimpleNamespace(
                        message=message,
                        finish_reason=choice_data.get("finish_reason"),
                        index=choice_data.get("index", 0)
                    )
                    choices.append(choice)
                
                return SimpleNamespace(
                    id=result.get("id"),
                    choices=choices,
                    usage=usage,
                    model=result.get("model"),
                    created=result.get("created")
                )
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"Kimi API HTTP error: {e.response.status_code} - {e.response.text}")
                raise HTTPException(e.response.status_code, f"Kimi API error: {e.response.text}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Kimi API response as JSON: {e}")
                logger.error(f"Response status: {response.status_code}, text: {response.text}")
                raise HTTPException(500, f"Invalid JSON response from Kimi API")
            except Exception as e:
                logger.error(f"Kimi API call failed: {e}", exc_info=True)
                raise
        else:
            # Kimi以外のモデルはLiteLLM使用
            logger.warning(f"Non-Kimi model {model}, falling back to LiteLLM")
            return await litellm.acompletion(**kwargs)
    
    async def acompletion(self, **kwargs) -> Any:
        """エイリアス"""
        return await self.completion(**kwargs)


# ============================================================================
# FastAPI アプリケーション
# ============================================================================

# グローバル状態
kimi_auth: Optional[KimiAuth] = None
kimi_config: Optional[KimiLiteLLMConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフスパン管理"""
    global kimi_auth, kimi_config
    
    # 起動時
    logger.info("Starting Claude Code Proxy with Kimi OAuth...")
    
    # Kimi認証初期化
    kimi_auth = KimiAuth(
        oauth_host=os.getenv("KIMI_OAUTH_HOST"),
        api_base_url=os.getenv("KIMI_BASE_URL"),
    )
    
    # ログイン済みの場合、自動更新を開始
    if kimi_auth.is_logged_in():
        logger.info("Already logged in to Kimi")
        kimi_auth.token_manager.start_auto_refresh(
            lambda rt: kimi_auth.refresh_token(rt)
        )
    else:
        # 自動ログイン機能
        auto_login = os.getenv("AUTO_LOGIN", "true").lower() == "true"
        auto_open_browser = os.getenv("AUTO_OPEN_BROWSER", "true").lower() == "true"
        
        if auto_login:
            logger.info("Starting automatic login...")
            try:
                async for event in kimi_auth.login(open_browser=auto_open_browser):
                    if event["type"] == "verification_url":
                        logger.info(f"Please visit: {event['data']['verification_url']}")
                        logger.info(f"User code: {event['data']['user_code']}")
                    elif event["type"] == "success":
                        logger.info("Login successful!")
                        break
                    elif event["type"] == "error":
                        logger.error(f"Login failed: {event['message']}")
                        logger.warning("Server will start without authentication. Use /auth/login to login manually.")
                        break
            except Exception as e:
                logger.error(f"Auto login failed: {e}")
                logger.warning("Server will start without authentication. Use /auth/login to login manually.")
        else:
            logger.warning("Auto login disabled. Please run /auth/login")
    
    # LiteLLM設定
    # ログ設定：CLI引数が最優先、次に環境変数
    log_conversations = CLI_LOG_CONVERSATIONS
    log_dir = CLI_LOG_DIR
    
    # CLI引数が設定されていない場合は環境変数をチェック
    if not CLI_LOG_CONVERSATIONS:
        log_conversations = os.getenv("KIMI_LOG_CONVERSATIONS", "false").lower() == "true"
    
    # ログディレクトリは環境変数も考慮
    if CLI_LOG_DIR == Path("logs/conversations"):
        log_dir = Path(os.getenv("KIMI_CONVERSATIONS_LOG_DIR", "logs/conversations"))
    
    kimi_config = KimiLiteLLMConfig(
        kimi_auth,
        log_conversations=log_conversations,
        log_dir=log_dir,
    )
    
    if log_conversations:
        logger.info(f"Conversation logging enabled: {log_dir}")
    
    yield
    
    # シャットダウン時
    logger.info("Shutting down...")
    if kimi_auth:
        kimi_auth.token_manager.stop_auto_refresh()
        await kimi_auth.close()


app = FastAPI(title="Claude Code Proxy with Kimi OAuth", lifespan=lifespan)


# ============================================================================
# 認証関連エンドポイント
# ============================================================================

@app.get("/auth/status")
async def auth_status():
    """認証状態を確認"""
    if not kimi_auth:
        return {"logged_in": False, "message": "Kimi auth not initialized"}
    
    token = kimi_auth.token_manager.token
    return {
        "logged_in": kimi_auth.is_logged_in(),
        "token_expires_at": token.expires_at if token else None,
        "token_expires_in": token.expires_in if token else None,
    }


@app.post("/auth/login")
async def auth_login(request: Request):
    """
    OAuthログインを開始
    
    Request body:
        - open_browser: bool (default: true)
    
    Response: SSEストリームで進捗を返す
    """
    if not kimi_auth:
        raise HTTPException(500, "Kimi auth not initialized")
    
    body = await request.json()
    open_browser = body.get("open_browser", True)
    
    async def event_generator():
        async for event in kimi_auth.login(open_browser=open_browser):
            yield f"data: {json.dumps(event)}\n\n"
        
        # 成功時は自動更新を開始
        if kimi_auth.is_logged_in():
            kimi_auth.token_manager.start_auto_refresh(
                lambda rt: kimi_auth.refresh_token(rt)
            )
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.post("/auth/logout")
async def auth_logout():
    """ログアウト"""
    if not kimi_auth:
        raise HTTPException(500, "Kimi auth not initialized")
    
    await kimi_auth.logout()
    return {"message": "Logged out successfully"}


@app.get("/auth/models")
async def auth_models():
    """利用可能なKimiモデル一覧"""
    if not kimi_auth:
        raise HTTPException(500, "Kimi auth not initialized")
    
    if not kimi_auth.is_logged_in():
        raise HTTPException(401, "Not logged in")
    
    try:
        models = await kimi_auth.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(500, f"Failed to list models: {e}")


# ============================================================================
# Anthropic API 互換エンドポイント
# ============================================================================

def convert_anthropic_to_openai(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Anthropic APIリクエストをOpenAI形式に変換"""
    messages = []
    
    # システムメッセージ
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        else:
            system_text = "\n\n".join(
                block.text for block in anthropic_request.system 
                if hasattr(block, "text")
            )
            if system_text:
                messages.append({"role": "system", "content": system_text})
    
    # 会話メッセージ
    for msg in anthropic_request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            # 複雑なコンテンツブロックの処理
            text_parts = []
            tool_calls = []
            
            for block in msg.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        # アシスタントのツール呼び出し
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            }
                        })
                    elif block.type == "tool_result":
                        # ツール結果は別のメッセージとして送信
                        content = getattr(block, "content", "")
                        if isinstance(content, str):
                            result_content = content
                        else:
                            result_content = json.dumps(content)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": result_content,
                        })
            
            # テキストとツール呼び出しを含むメッセージを追加
            if text_parts or tool_calls:
                msg_dict = {"role": msg.role}
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                if tool_calls and msg.role == "assistant":
                    msg_dict["tool_calls"] = tool_calls
                    # contentがない場合は空文字列を設定
                    if not text_parts:
                        msg_dict["content"] = ""
                # reasoning_contentは追加しない（kimi-cliではThinkPartがある場合のみ）
                if msg_dict.get("content") or msg_dict.get("tool_calls"):
                    messages.append(msg_dict)
    
    # OpenAIリクエスト構築
    openai_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": anthropic_request.max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    
    if anthropic_request.stop_sequences:
        openai_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        openai_request["top_p"] = anthropic_request.top_p
    
    # デバッグ: 送信前のメッセージを確認
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            logger.info(f"DEBUG: Message {idx} - role={msg.get('role')}, has_tool_calls={bool(msg.get('tool_calls'))}, has_reasoning_content={'reasoning_content' in msg}, reasoning_content={msg.get('reasoning_content', 'N/A')}")
    
    # ツール変換
    if anthropic_request.tools:
        openai_tools = []
        for tool in anthropic_request.tools:
            tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else tool
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": tool_dict.get("input_schema", {}),
                },
            })
        openai_request["tools"] = openai_tools
        logger.info(f"Converted {len(openai_tools)} tools for request")
        
        # tool_choiceの変換（Kimi APIはサポートしていない可能性があるため扱いに注意）
        if anthropic_request.tool_choice:
            if isinstance(anthropic_request.tool_choice, dict):
                tc_type = anthropic_request.tool_choice.get("type")
                # Kimi APIはtool_choiceをサポートしていない可能性があるため警告
                logger.warning(f"tool_choice requested but may not be supported by Kimi: {tc_type}")
    
    return openai_request


def convert_openai_to_anthropic(
    openai_response: Any,
    original_request: MessagesRequest,
) -> MessagesResponse:
    """OpenAIレスポンスをAnthropic形式に変換"""
    
    # LiteLLM ModelResponseからデータを抽出
    if hasattr(openai_response, "choices"):
        choices = openai_response.choices
        message = choices[0].message if choices else None
        content_text = message.content if message else ""
        tool_calls = message.tool_calls if message else None
        finish_reason = choices[0].finish_reason if choices else "stop"
        usage_info = openai_response.usage
        response_id = getattr(openai_response, "id", f"msg_{uuid.uuid4().hex[:24]}")
    else:
        # 辞書形式
        response_dict = openai_response if isinstance(openai_response, dict) else openai_response.model_dump()
        choices = response_dict.get("choices", [{}])
        message = choices[0].get("message", {}) if choices else {}
        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls")
        finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
        usage_info = response_dict.get("usage", {})
        response_id = response_dict.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    
    # コンテンツ構築
    content = []
    
    if content_text:
        content.append({"type": "text", "text": content_text})
    
    # ツール呼び出し
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = func.get("name", "")
                args = func.get("arguments", "{}")
            else:
                func = getattr(tc, "function", None)
                tool_id = getattr(tc, "id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = getattr(func, "name", "") if func else ""
                args = getattr(func, "arguments", "{}") if func else "{}"
            
            # JSONパース
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            
            content.append({
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": args,
            })
    
    if not content:
        content.append({"type": "text", "text": ""})
    
    # 使用状況
    if isinstance(usage_info, dict):
        prompt_tokens = usage_info.get("prompt_tokens", 0)
        completion_tokens = usage_info.get("completion_tokens", 0)
    else:
        prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
        completion_tokens = getattr(usage_info, "completion_tokens", 0)
    
    # stop_reasonマッピング
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")
    
    return MessagesResponse(
        id=response_id,
        model=original_request.model,
        role="assistant",
        content=content,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        ),
    )


async def convert_non_streaming_to_streaming(
    response: Any,
    original_request: MessagesRequest,
) -> AsyncIterator[str]:
    """非ストリーミングレスポンスをストリーミング形式に変換"""
    
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    # 使用状況を取得
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage") and response.usage:
        if hasattr(response.usage, "prompt_tokens"):
            input_tokens = response.usage.prompt_tokens
        if hasattr(response.usage, "completion_tokens"):
            output_tokens = response.usage.completion_tokens
    
    # message_start
    message_start_data = {
        'type': 'message_start',
        'message': {
            'id': message_id,
            'type': 'message',
            'role': 'assistant',
            'model': original_request.model,
            'content': [],
            'stop_reason': None,
            'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"
    
    # レスポンスからコンテンツとtool_callsを取得
    finish_reason = 'stop'  # デフォルト値
    
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or 'stop'
        
        content_index = 0
        
        # テキストコンテンツがあれば送信
        if hasattr(message, "content") and message.content:
            # content_block_start
            content_block_start_data = {
                'type': 'content_block_start',
                'index': content_index,
                'content_block': {'type': 'text', 'text': ''}
            }
            yield f"event: content_block_start\ndata: {json.dumps(content_block_start_data)}\n\n"
            
            # content_block_delta
            content_block_delta_data = {
                'type': 'content_block_delta',
                'index': content_index,
                'delta': {'type': 'text_delta', 'text': message.content}
            }
            yield f"event: content_block_delta\ndata: {json.dumps(content_block_delta_data)}\n\n"
            
            # content_block_stop
            content_block_stop_data = {
                'type': 'content_block_stop',
                'index': content_index
            }
            yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop_data)}\n\n"
            
            content_index += 1
        
        # tool_callsがあれば送信
        if hasattr(message, "tool_calls") and message.tool_calls:
            logger.info(f"Converting {len(message.tool_calls)} tool calls to streaming format")
            
            for tool_call in message.tool_calls:
                # content_block_start
                tool_block_start_data = {
                    'type': 'content_block_start',
                    'index': content_index,
                    'content_block': {
                        'type': 'tool_use',
                        'id': tool_call.id,
                        'name': tool_call.function.name,
                        'input': {}
                    }
                }
                yield f"event: content_block_start\ndata: {json.dumps(tool_block_start_data)}\n\n"
                
                # content_block_delta (input)
                tool_input = json.loads(tool_call.function.arguments)
                tool_block_delta_data = {
                    'type': 'content_block_delta',
                    'index': content_index,
                    'delta': {
                        'type': 'input_json_delta',
                        'partial_json': tool_call.function.arguments
                    }
                }
                yield f"event: content_block_delta\ndata: {json.dumps(tool_block_delta_data)}\n\n"
                
                # content_block_stop
                tool_block_stop_data = {
                    'type': 'content_block_stop',
                    'index': content_index
                }
                yield f"event: content_block_stop\ndata: {json.dumps(tool_block_stop_data)}\n\n"
                
                content_index += 1
    
    # message_delta
    stop_reason_map = {
        'stop': 'end_turn',
        'tool_calls': 'tool_use',
        'length': 'max_tokens',
    }
    anthropic_stop_reason = stop_reason_map.get(finish_reason, 'end_turn')
    
    message_delta_data = {
        'type': 'message_delta',
        'delta': {'stop_reason': anthropic_stop_reason, 'stop_sequence': None},
        'usage': {'output_tokens': output_tokens}
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta_data)}\n\n"
    
    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


async def handle_streaming(
    response_generator: AsyncIterator[Any],
    original_request: MessagesRequest,
) -> AsyncIterator[str]:
    """ストリーミングレスポンスを処理"""
    
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    # message_start
    message_start_data = {
        'type': 'message_start',
        'message': {
            'id': message_id,
            'type': 'message',
            'role': 'assistant',
            'model': original_request.model,
            'content': [],
            'stop_reason': None,
            'usage': {'input_tokens': 0, 'output_tokens': 0},
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"
    
    # content_block_start
    content_block_start_data = {
        'type': 'content_block_start',
        'index': 0,
        'content_block': {'type': 'text', 'text': ''}
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start_data)}\n\n"
    
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    
    accumulated_text = ""
    input_tokens = 0
    output_tokens = 0
    
    async for chunk in response_generator:
        # 使用状況
        if hasattr(chunk, "usage") and chunk.usage:
            if hasattr(chunk.usage, "prompt_tokens"):
                input_tokens = chunk.usage.prompt_tokens
            if hasattr(chunk.usage, "completion_tokens"):
                output_tokens = chunk.usage.completion_tokens
        
        # コンテンツ処理
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            
            if delta and hasattr(delta, "content") and delta.content:
                text = delta.content
                accumulated_text += text
                content_block_delta_data = {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {'type': 'text_delta', 'text': text}
                }
                yield f"event: content_block_delta\ndata: {json.dumps(content_block_delta_data)}\n\n"
    
    # content_block_stop
    content_block_stop_data = {
        'type': 'content_block_stop',
        'index': 0
    }
    yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop_data)}\n\n"
    
    # message_delta（使用状況）
    message_delta_data = {
        'type': 'message_delta',
        'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
        'usage': {'output_tokens': output_tokens}
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta_data)}\n\n"
    
    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    
    # 会話ログを保存（ストリーミング完了後）
    # Note: ログ記録はmessages()関数で行われます


@app.post("/v1/messages")
async def messages(request: MessagesRequest):
    """Anthropic Messages APIエンドポイント"""
    
    if not kimi_config:
        raise HTTPException(500, "Server not initialized")
    
    # 認証チェック
    if not kimi_auth or not kimi_auth.is_logged_in():
        raise HTTPException(
            401,
            detail="Not authenticated. Please login first by calling POST /auth/login or restart server with AUTO_LOGIN=true"
        )
    
    logger.info(f"Request model: {request.model}, stream: {request.stream}")
    
    # OpenAI形式に変換
    openai_request = convert_anthropic_to_openai(request)
    
    try:
        if request.stream:
            # ストリーミング（Kimi APIからは非ストリーミングで受信してストリーミング形式に変換）
            logger.info(f"Starting request to Kimi API...")
            response = await kimi_config.completion(**openai_request)
            
            # 非ストリーミングレスポンスをストリーミング形式に変換
            async def streaming_with_log():
                async for chunk in convert_non_streaming_to_streaming(response, request):
                    yield chunk
                
                # ストリーミング完了後にログを保存
                if kimi_config.log_conversations:
                    try:
                        # レスポンスオブジェクトから直接情報を取得
                        content_items = []
                        if hasattr(response, "choices") and response.choices:
                            message = response.choices[0].message
                            if hasattr(message, "content") and message.content:
                                content_items.append({"type": "text", "text": message.content})
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    tool_input = json.loads(tool_call.function.arguments)
                                    content_items.append({
                                        "type": "tool_use",
                                        "id": tool_call.id,
                                        "name": tool_call.function.name,
                                        "input": tool_input
                                    })
                        
                        input_tokens = response.usage.prompt_tokens if hasattr(response, "usage") else 0
                        output_tokens = response.usage.completion_tokens if hasattr(response, "usage") else 0
                        finish_reason = response.choices[0].finish_reason if hasattr(response, "choices") else "end_turn"
                        
                        stop_reason_map = {
                            'stop': 'end_turn',
                            'tool_calls': 'tool_use',
                            'length': 'max_tokens',
                        }
                        anthropic_stop_reason = stop_reason_map.get(finish_reason, 'end_turn')
                        
                        request_log = request.model_dump()
                        response_log = {
                            "id": f"msg_{uuid.uuid4().hex[:24]}",
                            "model": request.model,
                            "role": "assistant",
                            "content": content_items,
                            "stop_reason": anthropic_stop_reason,
                            "usage": {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                            }
                        }
                        save_conversation_log(request_log, response_log, kimi_config.log_dir, is_stream=True)
                    except Exception as log_error:
                        logger.warning(f"Failed to log streaming conversation: {log_error}")
            
            return StreamingResponse(
                streaming_with_log(),
                media_type="text/event-stream",
            )
        else:
            # 非ストリーミング
            logger.info(f"Starting non-streaming request to Kimi API...")
            response = await kimi_config.completion(**openai_request)
            logger.info(f"Received response from Kimi API, converting to Anthropic format...")
            anthropic_response = convert_openai_to_anthropic(response, request)
            
            # 会話ログを保存
            if kimi_config.log_conversations:
                try:
                    request_log = request.model_dump()
                    response_log = anthropic_response.model_dump()
                    save_conversation_log(request_log, response_log, kimi_config.log_dir, is_stream=False)
                except Exception as log_error:
                    logger.warning(f"Failed to log conversation: {log_error}")
            
            return anthropic_response
    
    except KimiAuthError as e:
        logger.error(f"Kimi auth error: {e}", exc_info=True)
        raise HTTPException(401, f"Authentication error: {e}")
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {e}")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: MessagesRequest):
    """トークンカウントエンドポイント（簡易実装）"""
    try:
        # 簡易的なトークン推定（1単語≒1.3トークン）
        total_tokens = 0
        
        # システムプロンプト
        if request.system:
            if isinstance(request.system, str):
                total_tokens += len(request.system.split()) * 1.3
            else:
                for block in request.system:
                    if hasattr(block, "text"):
                        total_tokens += len(block.text.split()) * 1.3
        
        # メッセージ
        for msg in request.messages:
            if isinstance(msg.content, str):
                total_tokens += len(msg.content.split()) * 1.3
            else:
                for block in msg.content:
                    if hasattr(block, "text"):
                        total_tokens += len(block.text.split()) * 1.3
        
        # ツール定義
        if request.tools:
            for tool in request.tools:
                tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else tool
                total_tokens += len(str(tool_dict).split()) * 1.3
        
        return {"input_tokens": int(total_tokens)}
    
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(500, f"Token counting error: {e}")


# ============================================================================
# ヘルスチェック
# ============================================================================

@app.get("/health")
async def health():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "ok",
        "kimi_logged_in": kimi_auth.is_logged_in() if kimi_auth else False,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/v1/models")
async def list_models():
    """
    Claude API互換のモデルリストエンドポイント
    利用可能なモデルのリストを返す
    """
    models = []
    
    # Kimiモデルを追加
    for model_id in KIMI_MODELS:
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(datetime(2024, 1, 1).timestamp()),
            "owned_by": "moonshot",
            "type": "chat"
        })
    
    # Claude互換のエイリアスを追加
    claude_models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620", 
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-sonnet-4-20250514"
    ]
    
    for model_id in claude_models:
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(datetime(2024, 1, 1).timestamp()),
            "owned_by": "anthropic",
            "type": "chat"
        })
    
    # OpenAIモデルを追加（フォールバック用）
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        for model_id in OPENAI_MODELS:
            models.append({
                "id": model_id,
                "object": "model",
                "created": int(datetime(2024, 1, 1).timestamp()),
                "owned_by": "openai",
                "type": "chat"
            })
    
    # Geminiモデルを追加（フォールバック用）
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        for model_id in GEMINI_MODELS:
            models.append({
                "id": model_id,
                "object": "model",
                "created": int(datetime(2024, 1, 1).timestamp()),
                "owned_by": "google",
                "type": "chat"
            })
    
    return {
        "object": "list",
        "data": models
    }


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "service": "Claude Code Proxy with Kimi OAuth",
        "version": "1.0.0",
        "endpoints": {
            "anthropic_api": "/v1/messages",
            "models": "/v1/models",
            "auth_status": "/auth/status",
            "auth_login": "/auth/login",
            "auth_logout": "/auth/logout",
            "auth_models": "/auth/models",
            "health": "/health",
        },
    }


# ============================================================================
# メインエントリーポイント
# ============================================================================

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Claude Code Proxy with Kimi OAuth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--logging",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Enable conversation logging (true/false)",
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/conversations",
        help="Directory to save conversation logs (default: logs/conversations)",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Server host (default: {DEFAULT_HOST})",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn
    
    # コマンドライン引数を解析
    args = parse_args()
    
    # グローバル設定を更新
    if args.logging is not None:
        CLI_LOG_CONVERSATIONS = args.logging.lower() == "true"
    CLI_LOG_DIR = Path(args.log_dir)
    
    # サーバー設定
    host = args.host or os.getenv("HOST", DEFAULT_HOST)
    port = args.port or int(os.getenv("PORT", DEFAULT_PORT))
    
    logger.info(f"Starting server on {host}:{port}")
    if CLI_LOG_CONVERSATIONS:
        logger.info(f"Conversation logging: ENABLED -> {CLI_LOG_DIR}")
    
    uvicorn.run(app, host=host, port=port)
