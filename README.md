# Claude Code Proxy with Kimi OAuth 🔐

**Claude CodeクライアントでKimi APIを使用できるプロキシサーバー**

OAuth Device Authorization Grantによる自動認証をサポートし、Kimiの強力なLLMをClaude Codeクライアントからシームレスに利用できます。

## ✨ 機能

- 🔐 **OAuth自動認証** - Device Authorization Grantフロー
- � **自動ログイン** - 初回起動時に自動的にログインフローを開始
- �🔄 **自動トークン更新** - 期限切れ前に自動的にリフレッシュ
- 🤖 **Kimi API対応** - Claude CodeからKimiモデルを直接使用
- 🌊 **ストリーミング対応** - リアルタイムレスポンス
- 🛠️ **ツール呼び出し** - function callingをサポート

## 🚀 クイックスタート

### 1. インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/claude-code-proxy-kimi.git
cd claude-code-proxy-kimi

# uvを使用して依存関係をインストール
uv sync

# または pip
cd claude-code-proxy-kimi
pip install -e .
```

### 2. 環境設定

```bash
# .envファイルをコピー
cp .env.example .env

# 編集は不要（OAuthで自動認証）
```

### 3. サーバー起動

**基本起動:**

```bash
# uvを使用
uv run python server.py

# または
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**会話ログを有効にして起動:**

```bash
# 会話ログを有効化
uv run python server.py --logging true

# ログディレクトリも指定
uv run python server.py --logging true --log-dir my_logs

# ホストとポートも指定
uv run python server.py --logging true --host 0.0.0.0 --port 8082
```

**コマンドライン引数:**

| 引数 | 値 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--logging` | `true`/`false` | 環境変数または `false` | 会話ログの記録 |
| `--log-dir` | ディレクトリパス | `logs/conversations` | ログ保存先 |
| `--host` | ホスト | `0.0.0.0` | サーバーホスト |
| `--port` | ポート番号 | `8082` | サーバーポート |

**🎉 初回起動時は自動的にOAuthログインが開始されます！**

サーバー起動時に以下の動作を行います：
- トークンが保存されていない場合、自動的にログインフローを開始
- デフォルトでブラウザが自動的に開き、認証ページに移動
- 認証完了後、自動的にトークンを保存

自動ログインを無効にする場合は、`.env`ファイルで設定：
```env
AUTO_LOGIN=false          # 自動ログインを無効化
AUTO_OPEN_BROWSER=false   # ブラウザ自動オープンを無効化
```

### 4. OAuthログイン（手動）

自動ログインを無効にした場合や、再ログインが必要な場合：

ブラウザで以下にアクセス：

**Unix/Linux/Mac:**
```bash
curl -X POST http://localhost:8082/auth/login \
  -H "Content-Type: application/json" \
  -d '{"open_browser": false}'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8082/auth/login -Method POST -ContentType "application/json" -Body '{"open_browser": false}'
```

またはブラウザを自動で開く：

**Unix/Linux/Mac:**
```bash
curl -X POST http://localhost:8082/auth/login \
  -H "Content-Type: application/json" \
  -d '{"open_browser": true}'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8082/auth/login -Method POST -ContentType "application/json" -Body '{"open_browser": true}'
```

表示されたURLをブラウザで開き、認証を完了してください。

### 5. Claude Codeで使用

**Unix/Linux/Mac:**
```bash
# プロキシを指定してClaude Codeを起動
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

**PowerShell:**
```powershell
# 環境変数を設定
$env:ANTHROPIC_BASE_URL="http://localhost:8082"

# Claude Codeを起動
claude
```

## 📋 APIエンドポイント

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/` | GET | サービス情報 |
| `/health` | GET | ヘルスチェック |
| `/auth/status` | GET | 認証状態確認 |
| `/auth/login` | POST | OAuthログイン開始 |
| `/auth/logout` | POST | ログアウト |
| `/auth/models` | GET | 利用可能なモデル一覧 |
| `/v1/messages` | POST | Anthropic Messages API |

## 🔧 設定

### 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `HOST` | `0.0.0.0` | サーバーホスト（`--host`で指定可） |
| `PORT` | `8082` | サーバーポート（`--port`で指定可） |
| `AUTO_LOGIN` | `true` | サーバー起動時に自動ログイン |
| `AUTO_OPEN_BROWSER` | `true` | 自動ログイン時にブラウザを自動で開く |
| `KIMI_LOG_CONVERSATIONS` | `false` | 会話ログを記録（`--logging`で指定可、優先度高） |
| `KIMI_CONVERSATIONS_LOG_DIR` | `logs/conversations` | 会話ログの保存先（`--log-dir`で指定可） |
| `KIMI_OAUTH_HOST` | `https://auth.kimi.com` | OAuthホスト |
| `KIMI_BASE_URL` | `https://api.kimi.com/coding/v1` | APIベースURL |
| `PREFERRED_PROVIDER` | `kimi` | 優先プロバイダー |
| `BIG_MODEL` | `kimi-k2-turbo-preview` | 大きいモデル |
| `SMALL_MODEL` | `kimi-k2-turbo-preview` | 小さいモデル |

### モデルマッピング

| Claudeモデル | マッピング先 |
|-------------|------------|
| `claude-3-opus` | `kimi-k2-turbo-preview` |
| `claude-3-sonnet` | `kimi-k2-turbo-preview` |
| `claude-3-haiku` | `kimi-k2-turbo-preview` |

### 利用可能なKimiモデル

- `kimi-k2-turbo-preview` - 標準モデル
- `kimi-k2.5` - 最新モデル
- `kimi-for-coding` - コーディング特化モデル

## 📝 使用例

### ログイン状態確認

**Unix/Linux/Mac:**
```bash
curl http://localhost:8082/auth/status
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8082/auth/status
```

### モデル一覧取得

**Unix/Linux/Mac:**
```bash
curl http://localhost:8082/auth/models
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8082/auth/models
```

### メッセージ送信（APIテスト）

**Unix/Linux/Mac:**
```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "こんにちは！"}]
  }'
```

**PowerShell:**
```powershell
$body = @'
{
  "model": "claude-3-sonnet",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "こんにちは！"}]
}
'@
Invoke-RestMethod -Uri http://localhost:8082/v1/messages -Method POST -ContentType "application/json" -Body $body
```

## 🔒 セキュリティ

- トークンは `~/.kimi/credentials/kimi-code.json` に保存されます
- ファイルパーミッションは `600` に設定されます
- トークンは自動的に更新され、手動での管理は不要です

## � 会話ログ機能

デバッグや分析のために、会話の内容をJSONファイルに記録できます。

### ログ機能を有効にする

**方法1: コマンドライン引数で指定（推奨）**

```bash
# 会話ログを有効化
uv run python server.py --logging true

# ログディレクトリも指定
uv run python server.py --logging true --log-dir my_logs
```

**方法2: 環境変数で設定**

`.env`ファイルで設定：

```env
# 会話ログを有効化
KIMI_LOG_CONVERSATIONS=true

# ログの保存先（オプション）
KIMI_CONVERSATIONS_LOG_DIR=logs/conversations
```

**方法3: プログラムコードで設定**

[server.py](claude-code-proxy-kimi/server.py)の先頭で、`CLI_LOG_CONVERSATIONS`グローバル変数を変更：

```python
# グローバル設定（コマンドライン引数から設定）
CLI_LOG_CONVERSATIONS = True  # ログを有効化
CLI_LOG_DIR = Path("logs/conversations")  # ログ保存先
```

**優先順位:** コマンドライン引数 > 環境変数 > デフォルト値

### ログファイル形式

会話ログは以下の形式で保存されます：

```json
{
  "timestamp": "2026-02-11T01:23:45.678901",
  "request": {
    "model": "claude-3-sonnet",
    "messages": [...],
    "max_tokens": 1024,
    "stream": false
  },
  "response": {
    "id": "msg_abc123...",
    "model": "claude-3-sonnet",
    "content": [...],
    "usage": {
      "input_tokens": 10,
      "output_tokens": 50
    }
  },
  "is_stream": false
}
```

ログファイル名: `conversation_20260211_012345_abc12345.json`

**⚠️ 注意**: 会話ログには機密情報が含まれる可能性があります。本番環境では使用せず、デバッグ時のみ有効にしてください。

## �🐛 トラブルシューティング

### ログインに失敗する

**Unix/Linux/Mac:**
```bash
# ログアウトして再ログイン
curl -X POST http://localhost:8082/auth/logout

# 手動でログイン
curl -X POST http://localhost:8082/auth/login -d '{"open_browser": false}'
```

**PowerShell:**
```powershell
# ログアウトして再ログイン
Invoke-RestMethod -Uri http://localhost:8082/auth/logout -Method POST

# 手動でログイン
Invoke-RestMethod -Uri http://localhost:8082/auth/login -Method POST -ContentType "application/json" -Body '{"open_browser": false}'
```

### トークン期限切れ

自動更新が有効になっていれば問題ありません。手動で更新する場合：

**Unix/Linux/Mac:**
```bash
curl http://localhost:8082/auth/status
curl -X POST http://localhost:8082/auth/login
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8082/auth/status
Invoke-RestMethod -Uri http://localhost:8082/auth/login -Method POST
```

### Claude Codeが接続できない

**Unix/Linux/Mac:**
```bash
# サーバーが起動しているか確認
curl http://localhost:8082/health

# 環境変数を確認
export ANTHROPIC_BASE_URL=http://localhost:8082
echo $ANTHROPIC_BASE_URL
```

**PowerShell:**
```powershell
# サーバーが起動しているか確認
Invoke-RestMethod -Uri http://localhost:8082/health

# 環境変数を確認
$env:ANTHROPIC_BASE_URL="http://localhost:8082"
echo $env:ANTHROPIC_BASE_URL
```

## 🏗️ アーキテクチャ

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  Claude Code    │────▶│  Proxy Server       │────▶│  Kimi API    │
│  Client         │◀────│  (This Project)     │◀────│  (OAuth)     │
└─────────────────┘     └─────────────────────┘     └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Token Store │
                        │  ~/.kimi/    │
                        └──────────────┘
```

## 📦 ファイル構成

```
claude-code-proxy-kimi/
├── server.py           # メインサーバー
├── kimi_auth.py        # OAuth認証モジュール
├── token_manager.py    # トークン管理
├── .env.example        # 環境変数例
├── pyproject.toml      # 依存関係
└── README.md           # このファイル
```

## 🤝 貢献

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 ライセンス

MIT License

## 🙏 謝辞

- 元の [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) プロジェクト
- [LiteLLM](https://github.com/BerriAI/litellm) - 統一LLMインターフェース
- [Kimi](https://kimi.com/) - 強力なLLM API
