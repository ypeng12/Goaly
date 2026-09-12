#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo " Starting Aegis ClaimShield Insurance SOP Agent"
echo "=========================================================="

# Run test suite to verify harness integrity
echo "Running automated test suite..."
python3 -m pytest tests/ -q

PORT="${PORT:-8080}"
echo "Tests passed! Starting FastAPI server on http://localhost:$PORT"
echo "You can open http://localhost:$PORT in your browser to test."
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT" --reload
