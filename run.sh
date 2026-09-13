#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
APP_PYTHON="${PYTHON:-python3}"
if [[ -z "${PYTHON:-}" && -x "$APP_DIR/.venv/bin/python" ]]; then
  APP_PYTHON="$APP_DIR/.venv/bin/python"
fi
APP_PORT="${PORT:-8080}"
printf 'Insurance SOP demo: http://localhost:%s\n' "$APP_PORT"
exec "$APP_PYTHON" -m uvicorn backend.app:app --host "${BIND_HOST:-127.0.0.1}" --port "$APP_PORT"
