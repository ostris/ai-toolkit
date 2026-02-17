#!/bin/bash
set -e

# Copy baked HuggingFace configs into runtime HF_HOME.
# The image bakes configs at /opt/hf-cache, but HF_HOME is a bind mount
# that hides baked-in files. cp -rn = no-clobber (never overwrites existing).
if [ -d /opt/hf-cache/hub ] && [ -n "$HF_HOME" ]; then
    mkdir -p "$HF_HOME/hub"
    # Use cp -a to preserve symlinks (HF cache uses snapshot→blob symlinks).
    # --no-clobber prevents overwriting files already present from the host mount.
    cp -a --no-clobber /opt/hf-cache/hub/. "$HF_HOME/hub/" 2>/dev/null || true
fi

exec "$@"
