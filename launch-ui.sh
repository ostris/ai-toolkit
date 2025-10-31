#!/usr/bin/env bash
set -euo pipefail

# Launch script for Ostris AI-Toolkit UI + worker
# - Ensures Python venv + requirements
# - Ensures Node deps + DB
# - Builds UI and starts Next.js UI + worker on ${PORT:-8675}

REPO_DIR="/home/alexis/ai-toolkit"
VENV_DIR="$REPO_DIR/venv"
UI_DIR="$REPO_DIR/ui"
PORT="${PORT:-8675}"

cd "$REPO_DIR"

# Python venv
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
# Install python deps (best-effort: continue if one problematic optional pkg fails)
# Note: using a temp requirements file to allow retries if a single package fails.
"$VENV_DIR/bin/python" - << 'PY' || true
import subprocess, sys
req = 'requirements.txt'
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req])
except subprocess.CalledProcessError as e:
    print(f"[WARN] pip install -r {req} failed with code {e.returncode}; continuing.")
PY

# Node/Next UI
cd "$UI_DIR"
# Prefer npm ci if lockfile present; fallback to npm install
if [ -f package-lock.json ]; then
  npm ci || npm install
else
  npm install
fi
# Initialize Prisma DB (SQLite by default)
npm run update_db
# Build and start UI + worker
npm run build
# Start worker + Next UI bound to localhost to avoid port conflicts with Tailscale
exec npx concurrently --restart-tries -1 --restart-after 1000 -n WORKER,UI \
  "node dist/cron/worker.js" \
  "next start --hostname 127.0.0.1 --port ${PORT}"
