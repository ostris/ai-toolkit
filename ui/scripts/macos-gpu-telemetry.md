# macOS GPU Telemetry

This collector is optional and only used on macOS.

It reads real-time values from `powermetrics` (root-required) and writes a JSON snapshot that the UI API (`/api/gpu`) can read.

## Why this exists

Apple Silicon does not expose NVIDIA-style telemetry through `nvidia-smi`, and direct unprivileged APIs often miss:

- GPU die temperature
- Fan RPM
- GPU clock speed
- GPU power draw

`powermetrics` can provide these, but it must run as root.

## Manual run

From `ui/`:

```bash
sudo npm run mac:gpu-telemetry
```

By default this writes:

```text
/tmp/ai-toolkit-mac-gpu-telemetry.json
```

You can override it:

```bash
sudo AI_TOOLKIT_MAC_GPU_TELEMETRY_PATH=/custom/path/gpu.json npm run mac:gpu-telemetry
```

If you override the path, set the same env var for the UI server process so `/api/gpu` reads the same file.

## Optional launchd service

Use the provided example plist as a template:

```text
scripts/com.ai-toolkit.macos-gpu-telemetry.plist.example
```

Edit placeholders (`__UI_DIR__`, `__NODE_PATH__`, and log paths), then load with:

```bash
sudo cp scripts/com.ai-toolkit.macos-gpu-telemetry.plist /Library/LaunchDaemons/
sudo launchctl load -w /Library/LaunchDaemons/com.ai-toolkit.macos-gpu-telemetry.plist
```

## Cross-platform behavior

- macOS: API uses this telemetry cache when available.
- Linux/Windows: existing `nvidia-smi` flow is unchanged.
