#!/bin/bash
# One-shot searchbox run: answer a prompt from a dataroom zip/folder for a budget of turns.
#
#   bash scripts/run.sh "your question" path/to/dataroom.zip 300000 ./out
#
# Requires: a running OpenAI-compatible model server at $LLAMA_URL (e.g. llama-server :8080),
# the `pi` CLI on PATH, and the .venv set up (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
set -a; [ -f .env ] && . ./.env; set +a

QUERY="${1:?usage: run.sh <query> <dataroom.zip|folder> [budget] [outdir]}"
DATAROOM="${2:?usage: run.sh <query> <dataroom.zip|folder> [budget] [outdir]}"
BUDGET="${3:-${TURN_BUDGET:-30}}"
OUT="${4:-./out/$(date +%Y%m%d-%H%M%S)}"

[ -x "$ROOT/.venv/bin/python" ] || { echo "ERROR: .venv missing (see README)" >&2; exit 1; }
command -v pi >/dev/null || { echo "ERROR: pi not found (npm i -g @earendil-works/pi-coding-agent)" >&2; exit 1; }

export PATH="$ROOT/.venv/bin:$(dirname "$(command -v pi)"):$PATH"
export PI_BIN="$(command -v pi)"
export PI_SKIP_VERSION_CHECK=1
export LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:8080}"

echo "query:  $QUERY"
echo "dataroom: $DATAROOM"
echo "budget: $BUDGET turns"
echo "out:    $OUT"
exec "$ROOT/.venv/bin/python" -m server.run_searchbox \
  --query "$QUERY" --dataroom "$DATAROOM" --budget "$BUDGET" --out "$OUT"
