"""
Kimi Token Manager

OAuthトークンの保存・読み込み・自動更新を管理します。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)

# 定数
REFRESH_THRESHOLD_SECONDS = 300  # 期限切れ5分前に更新
DEFAULT_CONFIG_DIR = Path.home() / ".kimi"
DEFAULT_CREDENTIALS_FILE = DEFAULT_CONFIG_DIR / "credentials" / "kimi-code.json"


@dataclass
class OAuthToken:
    """OAuthトークンデータクラス"""
    access_token: str
    refresh_token: str
    expires_at: float
    scope: str = ""
    token_type: str = "Bearer"
    
    @property
    def is_expired(self) -> bool:
        """トークンが期限切れかチェック"""
        return time.time() >= self.expires_at
    
    @property
    def needs_refresh(self) -> bool:
        """トークン更新が必要かチェック"""
        return self.expires_at - time.time() < REFRESH_THRESHOLD_SECONDS
    
    @property
    def expires_in(self) -> int:
        """残り有効秒数"""
        return max(0, int(self.expires_at - time.time()))
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthToken:
        return cls(
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            expires_at=float(data.get("expires_at", 0)),
            scope=data.get("scope", ""),
            token_type=data.get("token_type", "Bearer"),
        )


class TokenManager:
    """
    OAuthトークンを管理し、自動更新を行うクラス
    """
    
    def __init__(
        self,
        credentials_file: Optional[Path] = None,
        auto_refresh: bool = True,
        refresh_callback: Optional[Callable[[OAuthToken], None]] = None,
    ):
        self.credentials_file = credentials_file or DEFAULT_CREDENTIALS_FILE
        self.auto_refresh = auto_refresh
        self.refresh_callback = refresh_callback
        self._token: Optional[OAuthToken] = None
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._stop_refresh = asyncio.Event()
        
        # 初回読み込み
        self._load_token()
    
    @property
    def token(self) -> Optional[OAuthToken]:
        """現在のトークンを取得"""
        return self._token
    
    @property
    def is_logged_in(self) -> bool:
        """ログイン状態を確認"""
        return self._token is not None and not self._token.is_expired
    
    @property
    def access_token(self) -> Optional[str]:
        """アクセストークンを取得"""
        if self._token:
            return self._token.access_token
        return None
    
    def _ensure_dir(self) -> None:
        """設定ディレクトリが存在することを確認"""
        self.credentials_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_token(self) -> Optional[OAuthToken]:
        """ファイルからトークンを読み込み"""
        try:
            if self.credentials_file.exists():
                data = json.loads(self.credentials_file.read_text(encoding="utf-8"))
                self._token = OAuthToken.from_dict(data)
                logger.info("Token loaded from file")
                return self._token
        except Exception as e:
            logger.warning(f"Failed to load token: {e}")
        return None
    
    def save_token(self, token: OAuthToken) -> None:
        """トークンをファイルに保存"""
        try:
            self._ensure_dir()
            self.credentials_file.write_text(
                json.dumps(token.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # パーミッションを600に設定（可能な場合）
            try:
                os.chmod(self.credentials_file, 0o600)
            except Exception:
                pass
            self._token = token
            logger.info("Token saved to file")
            
            # コールバックがあれば実行
            if self.refresh_callback:
                try:
                    self.refresh_callback(token)
                except Exception as e:
                    logger.error(f"Refresh callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to save token: {e}")
            raise
    
    def delete_token(self) -> None:
        """トークンを削除（ログアウト）"""
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
            self._token = None
            logger.info("Token deleted")
        except Exception as e:
            logger.error(f"Failed to delete token: {e}")
    
    async def ensure_fresh_token(self, refresh_func: Callable[[str], Any]) -> OAuthToken:
        """
        トークンが新鮮であることを確認（必要に応じて更新）
        
        Args:
            refresh_func: リフレッシュ関数 (refresh_token: str) -> OAuthToken
        
        Returns:
            有効なOAuthToken
        """
        if not self._token:
            raise TokenManagerError("No token available. Please login first.")
        
        if not self._token.needs_refresh:
            return self._token
        
        # ロックを取得
        async with self._refresh_lock:
            # 再チェック（他のタスクが更新済みの可能性）
            if self._token and not self._token.needs_refresh:
                return self._token
            
            if not self._token or not self._token.refresh_token:
                raise TokenManagerError("No refresh token available")
            
            try:
                logger.info("Refreshing token...")
                new_token = await refresh_func(self._token.refresh_token)
                self.save_token(new_token)
                logger.info("Token refreshed successfully")
                return new_token
            except Exception as e:
                logger.error(f"Token refresh failed: {e}")
                # 認証エラーの場合はトークンを削除
                if "unauthorized" in str(e).lower() or "invalid" in str(e).lower():
                    self.delete_token()
                    raise TokenManagerError("Session expired. Please login again.")
                raise
    
    def start_auto_refresh(
        self,
        refresh_func: Callable[[str], Any],
        interval_seconds: float = 60.0,
    ) -> None:
        """
        自動トークン更新を開始
        
        Args:
            refresh_func: リフレッシュ関数
            interval_seconds: チェック間隔（秒）
        """
        if self._refresh_task and not self._refresh_task.done():
            return
        
        self._stop_refresh.clear()
        self._refresh_task = asyncio.create_task(
            self._auto_refresh_loop(refresh_func, interval_seconds)
        )
        logger.info("Auto refresh started")
    
    def stop_auto_refresh(self) -> None:
        """自動トークン更新を停止"""
        if self._refresh_task:
            self._stop_refresh.set()
            self._refresh_task.cancel()
            self._refresh_task = None
            logger.info("Auto refresh stopped")
    
    async def _auto_refresh_loop(
        self,
        refresh_func: Callable[[str], Any],
        interval_seconds: float,
    ) -> None:
        """自動更新ループ"""
        try:
            while not self._stop_refresh.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_refresh.wait(),
                        timeout=interval_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                
                if self._token and self._token.needs_refresh:
                    try:
                        await self.ensure_fresh_token(refresh_func)
                    except Exception as e:
                        logger.warning(f"Auto refresh failed: {e}")
        except asyncio.CancelledError:
            pass


class TokenManagerError(Exception):
    """TokenManager固有のエラー"""
    pass
