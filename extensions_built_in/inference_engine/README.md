# Inference Engine

A resident generation server that runs as an AI Toolkit job and serves every
registered model arch (image, video, audio). It hot-swaps models per request:
components the new model shares with the previous one (a text encoder, a
VAE) stay resident, the rest is freed and loaded. Progress and the raw
per-step latents stream back over the request's HTTP connection.

Design notes and status: `PLANNING.md` in this folder.

## Running

From the UI: **Generate** in the sidebar, pick a GPU, **Start engine**. The
engine shows up in the Queue as an `inference` job and runs until stopped.

Headless:

```yaml
# inference_engine.yaml
job: extension
config:
  name: inference_engine
  process:
    - type: InferenceEngine
      device: cuda
      engine:
        job_folder: output/inference_engine   # engine.json + outputs/ + assets/
        # port: 8866         # fixed port for external clients (default: ephemeral)
        # token: secret      # request token (default: random per launch)
        # default_model: {arch: zimage}   # warm load
```

```
python run.py inference_engine.yaml
```

The server binds `127.0.0.1` on an ephemeral port and writes
`<job_folder>/engine.json` (`host`, `port`, `token`, `pid`). The UI proxies
`/api/inference/*` to it; nothing else is exposed.

## HTTP API

All routes except `/health` need the token (`X-Engine-Token: <token>` or
`Authorization: Bearer <token>`).

| route | |
|---|---|
| `GET /health` | busy/queue, active model, VRAM, pool residents + hit/miss stats |
| `GET /models` | every arch with modality, defaults, control-image requirement (`toolkit/models/registry.py`) |
| `POST /generate` | body `{model, sample, stream}` → binary frame stream (below). `{"wait": true}` returns JSON when done instead |
| `POST /cancel/{request_id}` | cancel a queued/running request |
| `GET /queue` | queued + recent requests |
| `POST /unload` | drop the holder and every pooled component |
| `POST /assets?name=x.png` | raw-body upload; returns `{path}` usable as `sample.ctrl_img` |
| `GET /outputs/{relpath}` | fetch a produced file |

`model` = `ModelConfig` kwargs (`arch` required; `name_or_path`, `quantize`,
`qtype`, `quantize_te`, `qtype_te`, `low_vram`, `layer_offloading`,
`lora_path`, `extras_name_or_path`, ...). Registry defaults fill in what is
omitted. `sample` = `GenerateImageConfig` kwargs (`prompt`, `negative_prompt`,
`width`, `height`, `num_inference_steps`, `guidance_scale`, `seed`,
`num_frames`, `fps`, `ctrl_img`, `output_ext`). `stream` =
`{latents: "raw"|"none", every_n_steps, max_frames}`.

### Frame stream

`u32 header_len | json header | u64 payload_len | payload` (little endian).
Header always has `type` and `request_id`:

- `start` — resolved model/sample, `modality`, `preview` (latent→RGB factors)
- `status` — holder stage lines (loading, quantizing, ...)
- `progress` — `{step, total, elapsed}`
- `latent` — raw fp16 tensor payload; `shape`, `dtype`, `layout` (`BCHW`,
  `BCFHW`, `BLC`). Video latents are subsampled to `max_frames` frames.
- `result` — `{path, kind, ext, seed, width, height, ...}` per produced file
- `error` — `{message, cancelled}`
- `end` — `{status}`

Python reader: `protocol.FrameReader`; browser reader:
`ui/src/lib/engineStream.ts` (also does the latent→RGB projection from the
`preview` block, so the client needs no per-arch code).

### Per-step latents

`BaseModel.generate_images` wraps `scheduler.step` on the sampling scheduler
while `sample_step_hook` is set, so every diffusers-style pipeline reports
its latents without per-arch code. Holders with a hand-rolled loop call
`self._emit_sample_step(latents, i, n)` (ace_step does).

## Component pool

`toolkit/models/v2/pool.py`. While a pool is active, `OstrisModelMixin.
load_model` returns the resident module for an identical source request and
`aitk_post_load` is a no-op for a module already carrying that policy. After
each holder build the engine frees every pooled component the new model did
not touch. On CUDA OOM during a load it frees the untouched components and
retries once. Components loaded outside the mixin's `load_model` (custom
holder paths, legacy monolith archs) are not pooled and reload each time.

## Tests

```
python testing/test_inference_engine.py --arch zimage --quantize
python testing/test_inference_engine.py --arch zimage --cancel
python testing/test_inference_engine.py --swap zimage,zimage_l2p,zimage --quantize
python testing/test_inference_engine.py --all
```

`--quantize` forces convrot8 on transformer + TE for small GPUs;
`--allow-download` permits hub downloads (default offline → SKIP).
