"""
Kimi OAuth Authentication Module

Device Authorization Grantフローを実装したKimi認証モジュール
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, AsyncIterator

import httpx

from token_manager import OAuthToken, TokenManager

logger = logging.getLogger(__name__)

# 定数
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
DEFAULT_API_BASE_URL = "https://api.kimi.com/coding/v1"
KIMI_CODE_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_CONFIG_DIR = Path.home() / ".kimi"
DEVICE_ID_FILE = DEFAULT_CONFIG_DIR / "device_id"


@dataclass
class DeviceAuthorization:
    """デバイス認可レスポンス"""
    user_code: str
    device_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


class KimiAuthError(Exception):
    """Kimi認証エラー"""
    pass


class KimiAuthUnauthorized(KimiAuthError):
    """認証失敗（401/403）"""
    pass


class KimiDeviceExpired(KimiAuthError):
    """デバイスコード期限切れ"""
    pass


class KimiAuth:
    """
    Kimi OAuth認証クラス
    
    Device Authorization Grantフローを実装
    """
    
    def __init__(
        self,
        oauth_host: Optional[str] = None,
        api_base_url: Optional[str] = None,
        token_manager: Optional[TokenManager] = None,
        config_dir: Optional[Path] = None,
    ):
        self.oauth_host = oauth_host or os.getenv("KIMI_OAUTH_HOST") or DEFAULT_OAUTH_HOST
        self.api_base_url = api_base_url or os.getenv("KIMI_BASE_URL") or DEFAULT_API_BASE_URL
        self.token_manager = token_manager or TokenManager()
        self.config_dir = config_dir or DEFAULT_CONFIG_DIR
        self.device_id_file = self.config_dir / "device_id"
        self._device_id: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTPクライアントを取得（遅延初期化）"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def _get_or_create_device_id(self) -> str:
        """デバイスIDを取得または作成"""
        if self._device_id:
            return self._device_id
        
        try:
            if self.device_id_file.exists():
                self._device_id = self.device_id_file.read_text(encoding="utf-8").strip()
                return self._device_id
        except Exception:
            pass
        
        # 新規作成
        self._device_id = uuid.uuid4().hex
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.device_id_file.write_text(self._device_id, encoding="utf-8")
            os.chmod(self.device_id_file, 0o600)
        except Exception as e:
            logger.warning(f"Failed to save device_id: {e}")
        
        return self._device_id
    
    def _get_common_headers(self) -> dict[str, str]:
        """共通ヘッダーを生成"""
        system = platform.system()
        machine = platform.machine() or ""
        
        if system == "Darwin":
            device_model = f"macOS {platform.mac_ver()[0]} {machine}"
        elif system == "Windows":
            release = platform.release()
            device_model = f"Windows {release} {machine}"
        else:
            device_model = f"{system} {platform.release()} {machine}"
        
        return {
            "User-Agent": "KimiCLI/1.0.0 (Claude Code Proxy)",
            "X-Msh-Platform": "kimi_cli",
            "X-Msh-Version": "1.0.0",
            "X-Msh-Device-Name": platform.node() or socket.gethostname() or "unknown",
            "X-Msh-Device-Model": device_model,
            "X-Msh-Os-Version": platform.version(),
            "X-Msh-Device-Id": self._get_or_create_device_id(),
        }
    
    async def request_device_authorization(self) -> DeviceAuthorization:
        """
        デバイス認可をリクエスト
        
        Returns:
            DeviceAuthorization: 認可情報（verification_uri_completeなど）
        """
        client = await self._get_client()
        
        headers = self._get_common_headers()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        data = {"client_id": KIMI_CODE_CLIENT_ID}
        
        try:
            response = await client.post(
                f"{self.oauth_host}/api/oauth/device_authorization",
                data=data,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            
            return DeviceAuthorization(
                user_code=data["user_code"],
                device_code=data["device_code"],
                verification_uri=data.get("verification_uri", ""),
                verification_uri_complete=data["verification_uri_complete"],
                expires_in=int(data.get("expires_in", 0) or 0),
                interval=int(data.get("interval", 5)),
            )
        except httpx.HTTPStatusError as e:
            raise KimiAuthError(f"Device authorization failed: {e.response.text}")
        except Exception as e:
            raise KimiAuthError(f"Device authorization request failed: {e}")
    
    async def _request_device_token(
        self, auth: DeviceAuthorization
    ) -> tuple[int, dict[str, Any]]:
        """
        デバイストークンをポーリング
        
        Returns:
            (status_code, response_data)
        """
        client = await self._get_client()
        
        headers = self._get_common_headers()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        data = {
            "client_id": KIMI_CODE_CLIENT_ID,
            "device_code": auth.device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        
        try:
            response = await client.post(
                f"{self.oauth_host}/api/oauth/token",
                data=data,
                headers=headers,
            )
            return response.status_code, response.json()
        except httpx.HTTPError as e:
            raise KimiAuthError(f"Token polling request failed: {e}")
    
    async def refresh_token(self, refresh_token: str) -> OAuthToken:
        """
        リフレッシュトークンで新しいアクセストークンを取得
        
        Args:
            refresh_token: リフレッシュトークン
        
        Returns:
            OAuthToken: 新しいトークン
        """
        client = await self._get_client()
        
        headers = self._get_common_headers()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        data = {
            "client_id": KIMI_CODE_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        
        try:
            response = await client.post(
                f"{self.oauth_host}/api/oauth/token",
                data=data,
                headers=headers,
            )
            
            if response.status_code in (401, 403):
                raise KimiAuthUnauthorized(
                    response.json().get("error_description", "Token refresh unauthorized")
                )
            
            if response.status_code != 200:
                raise KimiAuthError(
                    response.json().get("error_description", "Token refresh failed")
                )
            
            data = response.json()
            return OAuthToken(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=data.get("expires_at") or (time.time() + data["expires_in"]),
                scope=data.get("scope", ""),
                token_type=data.get("token_type", "Bearer"),
            )
        except httpx.HTTPError as e:
            raise KimiAuthError(f"Token refresh request failed: {e}")
    
    async def login(
        self,
        open_browser: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        OAuthログインフローを実行
        
        Args:
            open_browser: ブラウザを自動で開くかどうか
        
        Yields:
            イベントオブジェクト {type, message, data?}
            
        使用例:
            async for event in kimi_auth.login():
                print(event["type"], event["message"])
        """
        import webbrowser
        
        yield {"type": "info", "message": "Starting OAuth login flow..."}
        
        while True:
            try:
                auth = await self.request_device_authorization()
            except Exception as e:
                yield {"type": "error", "message": f"Login failed: {e}"}
                raise
            
            yield {
                "type": "info",
                "message": "Please visit the following URL to finish authorization.",
            }
            
            yield {
                "type": "verification_url",
                "message": f"Verification URL: {auth.verification_uri_complete}",
                "data": {
                    "verification_url": auth.verification_uri_complete,
                    "user_code": auth.user_code,
                },
            }
            
            # ブラウザを開く
            if open_browser:
                try:
                    webbrowser.open(auth.verification_uri_complete)
                except Exception as e:
                    yield {
                        "type": "warning",
                        "message": f"Failed to open browser: {e}",
                    }
            
            # ポーリング
            interval_ms = max(auth.interval, 1) * 1000
            printed_wait = False
            
            try:
                while True:
                    status, data = await self._request_device_token(auth)
                    
                    if status == 200 and "access_token" in data:
                        token = OAuthToken(
                            access_token=data["access_token"],
                            refresh_token=data["refresh_token"],
                            expires_at=time.time() + data["expires_in"],
                            scope=data.get("scope", ""),
                            token_type=data.get("token_type", "Bearer"),
                        )
                        self.token_manager.save_token(token)
                        yield {"type": "success", "message": "Logged in successfully."}
                        return
                    
                    error_code = str(data.get("error") or "unknown_error")
                    
                    if error_code == "expired_token":
                        raise KimiDeviceExpired("Device code expired")
                    
                    error_description = str(data.get("error_description") or "")
                    
                    if not printed_wait:
                        yield {
                            "type": "waiting",
                            "message": "Waiting for user authorization...",
                            "data": {
                                "error": error_code,
                                "error_description": error_description,
                            },
                        }
                        printed_wait = True
                    
                    await asyncio.sleep(interval_ms / 1000)
                    
            except KimiDeviceExpired:
                yield {"type": "info", "message": "Device code expired, restarting login..."}
                continue
            except Exception as e:
                yield {"type": "error", "message": f"Login failed: {e}"}
                raise
    
    async def logout(self) -> None:
        """ログアウト（トークン削除）"""
        self.token_manager.delete_token()
        logger.info("Logged out")
    
    async def ensure_fresh_token(self) -> str:
        """
        有効なアクセストークンを取得（必要に応じて更新）
        
        Returns:
            str: アクセストークン
        """
        token = await self.token_manager.ensure_fresh_token(
            lambda rt: self.refresh_token(rt)
        )
        return token.access_token
    
    def is_logged_in(self) -> bool:
        """ログイン状態を確認"""
        return self.token_manager.is_logged_in
    
    async def list_models(self) -> list[dict[str, Any]]:
        """
        利用可能なモデル一覧を取得
        
        Returns:
            list[dict]: モデル情報リスト
        """
        access_token = await self.ensure_fresh_token()
        client = await self._get_client()
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            **self._get_common_headers(),
        }
        
        try:
            response = await client.get(
                f"{self.api_base_url}/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except httpx.HTTPStatusError as e:
            raise KimiAuthError(f"Failed to list models: {e.response.text}")
        except Exception as e:
            raise KimiAuthError(f"Failed to list models: {e}")
    
    async def close(self) -> None:
        """リソースを解放"""
        if self._client:
            await self._client.aclose()
            self._client = None
