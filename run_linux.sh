#!/usr/bin/env bash
# Update-and-run script for Linux — thin bootstrap over the in-repo manager.
#
# Everything (venv via uv-managed Python, torch for your GPU, requirements,
# portable Node.js and FFmpeg, dependency updates) is handled by
# `python -m manager`; this script only makes sure uv + a Python interpreter
# exist, then delegates. Works on desktop and headless boxes alike.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Banner ─────────────────────────────────────────────────────────
echo ""
printf '\033[36m'
cat << 'BANNER'
     _     ___   _____               _  _     _  _
    / \   |_ _| |_   _|  ___    ___ | || | __(_)| |_
   / _ \   | |    | |   / _ \  / _ \| || |/ /| || __|
  / ___ \  | |    | |  | (_) || (_) | ||   < | || |_
 /_/   \_\|___|   |_|   \___/  \___/|_||_|\_\|_| \__|
BANNER
printf '\033[0m'
printf '\033[90m  AI Toolkit Manager — Linux\033[0m\n'
echo ""

# ── 1. Ensure uv (prebuilt static binary, kept inside the repo) ─────
export PATH="$SCRIPT_DIR/.uv:$PATH"
export UV_PYTHON_INSTALL_DIR="$SCRIPT_DIR/.uv/python"
if ! command -v uv >/dev/null 2>&1; then
    echo "Downloading uv (package/python manager) into .uv/ ..."
    curl -LsSf https://astral.sh/uv/install.sh | \
        UV_INSTALL_DIR="$SCRIPT_DIR/.uv" UV_NO_MODIFY_PATH=1 sh
fi

# ── 2. Find a Python to run the manager (stdlib-only, needs >= 3.9) ─
find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON="$(find_python || true)"
if [[ -z "$PYTHON" ]]; then
    echo "No system Python found — provisioning one with uv..."
    uv python install 3.12
    PYTHON="$(uv python find 3.12)"
fi

# ── 3. Sync the environment and start the UI ────────────────────────
cd "$SCRIPT_DIR"
"$PYTHON" -m manager update --auto
exec "$PYTHON" -m manager launch
