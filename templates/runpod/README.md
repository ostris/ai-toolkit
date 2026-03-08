# RunPod Template (BIG DADDY VERSION)

This folder contains a RunPod-ready image template for this fork.

## What it does

- Builds an image from this repository (`ArtDesignAwesome/ai-toolkit_BIG-DADDY-VERSION`).
- Pre-installs Python dependencies and builds the UI during image build.
- Starts SSH (if `PUBLIC_KEY` is provided) and launches the UI on port `8675`.
- Persists `config`, `datasets`, `output`, and `aitk_db.db` under `/workspace/ai-toolkit-data`.

## Build and push image

From the repository root:

```bash
docker build -f templates/runpod/Dockerfile -t <your-docker-user>/aitk-big-daddy-runpod:latest templates/runpod --build-arg AITK_REF=main
docker push <your-docker-user>/aitk-big-daddy-runpod:latest
```

If you need a different branch or tag:

```bash
docker build -f templates/runpod/Dockerfile -t <your-docker-user>/aitk-big-daddy-runpod:<tag> templates/runpod --build-arg AITK_REF=<branch-or-tag>
docker push <your-docker-user>/aitk-big-daddy-runpod:<tag>
```

## RunPod template setup

Use your pushed image in a RunPod Pod template:

- Container image: `<your-docker-user>/aitk-big-daddy-runpod:latest`
- Exposed HTTP port: `8675`
- Volume mount: keep `/workspace` persistent (recommended)

Environment variables:

- Required: `AI_TOOLKIT_AUTH`
- Optional: `PUBLIC_KEY` (enables SSH key login)
- Optional: `AITK_UPDATE_ON_START=1` + `AITK_REF=<ref>` to pull latest code at pod startup

Use [`templates/runpod/.env.example`](./.env.example) as a baseline.

## 5090 / RTX Pro 6000 behavior

You do not set separate UI toggles for those cards. Throughput selection is automatic when train configs use:

```yaml
train:
  throughput_profile: auto
```

`auto` maps from device capability:

- 5090-class -> `ltx23_max`
- RTX Pro 6000-class (very high VRAM) -> `ltx23_ultra_vram`

Manual override is still available via `throughput_profile` in your train config.
