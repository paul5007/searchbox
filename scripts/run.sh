#!/bin/bash
# One-shot searchbox run: answer a prompt from a corpus zip/folder, spending a token budget.
#
#   bash scripts/run.sh "your question" path/to/corpus.zip 300000 ./out
#
# Requires: a running OpenAI-compatible model server at $LLAMA_URL (e.g. llama-server :8080),
# the `pi` CLI on PATH, and the .venv set up (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
set -a; [ -f .env ] && . ./.env; set +a

QUERY="${1:?usage: run.sh <query> <corpus.zip|folder> [budget] [outdir]}"
CORPUS="${2:?usage: run.sh <query> <corpus.zip|folder> [budget] [outdir]}"
BUDGET="${3:-${INPUT_TOKEN_BUDGET:-500000}}"
OUT="${4:-./out/$(date +%Y%m%d-%H%M%S)}"

[ -x "$ROOT/.venv/bin/python" ] || { echo "ERROR: .venv missing (see README)" >&2; exit 1; }
command -v pi >/dev/null || { echo "ERROR: pi not found (npm i -g @earendil-works/pi-coding-agent)" >&2; exit 1; }

export PATH="$ROOT/.venv/bin:$(dirname "$(command -v pi)"):$PATH"
export PI_BIN="$(command -v pi)"
export PI_SKIP_VERSION_CHECK=1
export LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:8080}"

echo "query:  $QUERY"
echo "corpus: $CORPUS"
echo "budget: $BUDGET input tokens"
echo "out:    $OUT"
exec "$ROOT/.venv/bin/python" -m server.run_searchbox \
  --query "$QUERY" --corpus "$CORPUS" --budget "$BUDGET" --out "$OUT"
