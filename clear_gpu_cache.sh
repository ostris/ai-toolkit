#!/usr/bin/env bash
set -e

echo "Clearing torch/triton GPU caches..."
rm -rf /tmp/torchinductor_*
rm -rf ~/.cache/torch/inductor/
rm -rf ~/.triton/cache/
echo "Done."
