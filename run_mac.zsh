#!/usr/bin/env zsh
# Update-and-run script for macOS — portable Python 3.12 + PyTorch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Banner ─────────────────────────────────────────────────────────
echo ""
echo "\033[36m"
cat << 'BANNER'
     _     ___   _____               _  _     _  _
    / \   |_ _| |_   _|  ___    ___ | || | __(_)| |_
   / _ \   | |    | |   / _ \  / _ \| || |/ /| || __|
  / ___ \  | |    | |  | (_) || (_) | ||   < | || |_
 /_/   \_\|___|   |_|   \___/  \___/|_||_|\_\|_| \__|
BANNER
echo "\033[0m"
echo "\033[90m  macOS Setup & Launcher\033[0m"
echo ""
VENV_DIR="$SCRIPT_DIR/.venv"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python3"
PYTHON_VERSION="3.12.8"
RELEASE_TAG="20241219"

# --- Package versions (update these as needed) ---
NODE_VERSION="23.11.1"
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCHAUDIO_VERSION="2.11.0"

# Detect architecture
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
    PLATFORM="aarch64-apple-darwin"
elif [[ "$ARCH" == "x86_64" ]]; then
    PLATFORM="x86_64-apple-darwin"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

# ── 1. Download standalone Python if needed ─────────────────────────
PYTHON_DIR="$SCRIPT_DIR/.python"
PYTHON_BIN="$PYTHON_DIR/bin/python3"

if [[ ! -x "$PYTHON_BIN" ]]; then
    TARBALL="cpython-${PYTHON_VERSION}+${RELEASE_TAG}-${PLATFORM}-install_only.tar.gz"
    URL="https://github.com/indygreg/python-build-standalone/releases/download/${RELEASE_TAG}/${TARBALL}"

    TMPDIR_DL="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_DL"' EXIT

    echo "Downloading standalone Python ${PYTHON_VERSION} (${PLATFORM})..."
    curl -fSL --progress-bar -o "$TMPDIR_DL/$TARBALL" "$URL"

    echo "Extracting..."
    tar -xzf "$TMPDIR_DL/$TARBALL" -C "$TMPDIR_DL"

    # Move to permanent location (the archive extracts to a "python" folder)
    rm -rf "$PYTHON_DIR"
    mv "$TMPDIR_DL/python" "$PYTHON_DIR"

    rm -rf "$TMPDIR_DL"
    trap - EXIT

    echo "Standalone Python installed to $PYTHON_DIR"
fi

# ── 2. Create venv if it doesn't exist ──────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

# ── 3. Download / update portable Node.js ──────────────────────────
NODE_DIR="$SCRIPT_DIR/.node"
NODE_BIN="$NODE_DIR/bin/node"

NEED_NODE=false
if [[ ! -x "$NODE_BIN" ]]; then
    NEED_NODE=true
elif [[ "$("$NODE_BIN" --version 2>/dev/null)" != "v${NODE_VERSION}" ]]; then
    echo "Node.js version mismatch (want v${NODE_VERSION}, have $("$NODE_BIN" --version))."
    NEED_NODE=true
fi

if $NEED_NODE; then
    if [[ "$ARCH" == "arm64" ]]; then
        NODE_ARCH="arm64"
    else
        NODE_ARCH="x64"
    fi

    NODE_TARBALL="node-v${NODE_VERSION}-darwin-${NODE_ARCH}.tar.gz"
    NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/${NODE_TARBALL}"

    TMPDIR_DL="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_DL"' EXIT

    echo "Downloading Node.js v${NODE_VERSION} (darwin-${NODE_ARCH})..."
    curl -fSL --progress-bar -o "$TMPDIR_DL/$NODE_TARBALL" "$NODE_URL"

    echo "Extracting..."
    tar -xzf "$TMPDIR_DL/$NODE_TARBALL" -C "$TMPDIR_DL"

    rm -rf "$NODE_DIR"
    mv "$TMPDIR_DL/node-v${NODE_VERSION}-darwin-${NODE_ARCH}" "$NODE_DIR"

    rm -rf "$TMPDIR_DL"
    trap - EXIT

    echo "Node.js v${NODE_VERSION} installed to $NODE_DIR"
else
    echo "Node.js v${NODE_VERSION} is up to date."
fi

# ── 4. Install / update PyTorch packages ────────────────────────────
# Helper: returns 0 if the package is installed at the exact version
pkg_ok() {
    local pkg="$1" want="$2"
    local got
    got="$("$PIP" show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')" || true
    [[ "$got" == "$want" ]]
}

PKGS_TO_INSTALL=()

pkg_ok "torch"       "$TORCH_VERSION"       || PKGS_TO_INSTALL+=("torch==$TORCH_VERSION")
pkg_ok "torchvision" "$TORCHVISION_VERSION"  || PKGS_TO_INSTALL+=("torchvision==$TORCHVISION_VERSION")
pkg_ok "torchaudio"  "$TORCHAUDIO_VERSION"   || PKGS_TO_INSTALL+=("torchaudio==$TORCHAUDIO_VERSION")

if (( ${#PKGS_TO_INSTALL[@]} )); then
    echo "Installing / updating: ${PKGS_TO_INSTALL[*]}"
    "$PIP" install "${PKGS_TO_INSTALL[@]}"
else
    echo "PyTorch packages are up to date."
fi

# ── 5. Install / update requirements.txt ────────────────────────────
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
REQ_HASH_FILE="$VENV_DIR/.requirements_hash"

if [[ -f "$REQUIREMENTS" ]]; then
    # Hash all requirements files (follows -r includes)
    CURRENT_HASH="$(cat "$SCRIPT_DIR"/requirements*.txt 2>/dev/null | shasum -a 256 | awk '{print $1}')"
    STORED_HASH=""
    [[ -f "$REQ_HASH_FILE" ]] && STORED_HASH="$(cat "$REQ_HASH_FILE")"

    if [[ "$CURRENT_HASH" != "$STORED_HASH" ]]; then
        echo "Installing / updating requirements.txt..."
        "$PIP" install -r "$REQUIREMENTS"
        echo "$CURRENT_HASH" > "$REQ_HASH_FILE"
    else
        echo "Requirements are up to date."
    fi
fi

# ── 6. Build and start the UI ───────────────────────────────────────
export PATH="$NODE_DIR/bin:$VENV_DIR/bin:$PATH"

echo ""
echo "Starting UI..."
cd "$SCRIPT_DIR/ui"
npm run build_and_start
