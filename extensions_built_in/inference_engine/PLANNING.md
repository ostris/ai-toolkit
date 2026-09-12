# Inference Engine — Planning

## Goal

A resident generation server that runs as a normal AI Toolkit job (like a
captioner job: `job: extension`, `process[0].type: InferenceEngine`), stays up
until stopped from the UI, and serves generation requests for every arch the
toolkit knows (image, video, audio). Each request names its model config; the
engine loads/reloads only the components that differ from what is resident,
drops what the request does not need, and streams progress + per-step latents
back over the request's connection, ending with the finished file.

This is Phase 3 of `toolkit/models/v2/PLANNING.md` ("live server with model
hot-swap"); everything below builds on the unified v2 load API
(`ModelClass.load(...)`, `aitk_post_load`, `component_load_kwargs`) that is
already in place for every non-legacy arch.

## Survey (2026-09-10)

What exists and is reused as-is:

- **Job plumbing**: `jobs/ExtensionJob.py` -> `get_all_extensions_process_dict`
  -> our process class. The UI launches jobs via `ui/cron/actions/startJob.ts`
  (`python run.py <config>` with `AITK_JOB_ID`, `CUDA_VISIBLE_DEVICES`,
  detached, stdout to the job log). Stop = `Job.stop` flag in sqlite, which
  `BaseCaptioner` polls in a daemon thread and turns into a SIGINT. The DB
  status/step/info writers (`_update_status`, `_update_key`, retry-on-lock,
  stop watcher) live in `extensions_built_in/captioner/BaseCaptioner.py`.
- **Generation path**: `BaseModel.generate_images([GenerateImageConfig])` does
  prompt encoding, control images, LoRA/network handling, then
  `generate_single_image` (per arch), then `gen_config.save_image_atomic`
  (png/jpg/webp image, animated webp video, wav/mp3 audio; `toolkit/audio/
  make_video.py` has the mp4 writer). `testing/test_model_loading.py` proves
  this flow for 34 archs and holds realistic per-arch sample settings.
- **Per-step hook point**: every arch samples through a diffusers-style
  scheduler object owned by the generation pipeline (`pipeline.scheduler.step`
  — diffusers pipelines, the vendored wan22/ltx/minimax/ace pipelines, the
  legacy monolith). Wrapping that one method yields (step, timestep, latents)
  for all of them without touching per-arch code.
- **Component loading**: `OstrisModelMixin.load` / `load_model` is the single
  funnel every v2 component goes through (DiTs, TEs, VAEs, vision encoders,
  connectors, vocoders). Legacy monolith archs adopt their pipeline
  components into v2 wrappers at one post-load funnel (`adopt_component`).
- **UI front**: `ui/cron/fileServer.ts` owns the public port and already
  spawns Next.js on an ephemeral loopback port and proxies to it (incl.
  websocket upgrades). Next.js route handlers stream `Response` bodies (the
  `/api/monitor` SSE route is the pattern). Bearer auth is a middleware on
  `/api/*`. `/api/files/*` serves job output folders. The job detail page
  already has a per-job `plugin.html` iframe tab.
- **Latent previews**: ComfyUI's `comfy/latent_formats.py` (read-only
  reference) has per-family linear `latent_rgb_factors` tables for SD1/SDXL/
  SD3/Flux/Flux2/Wan21/Wan22/LTXV/LTXAV/MiniMaxH3/Cosmos/HunyuanVideo.
- **HTTP server libs**: `fastapi` 0.140 and `uvicorn` 0.51 are importable in
  this env but only as transitive deps of gradio; they are NOT pinned in
  `requirements.txt`.

## Decisions (proposed — confirm before build)

1. **Transport = one chunked HTTP response per generation request** (not
   WebSocket). A `POST /generate` stays open for the whole generation and
   streams binary frames; the Next.js proxy route passes the body through as
   a `ReadableStream`, so it works with or without the fileServer front, in
   docker, on Windows/macOS, and from `curl`. WebSocket would require the
   fileServer to be mandatory (Next route handlers cannot upgrade). Cancel is
   a separate `POST /cancel/{request_id}` (also fired when the client aborts
   the stream).
2. **Frame format**: length-prefixed binary frames
   `u32 header_len | json header | u64 payload_len | payload`. Header
   `{type, request_id, ...}`; types: `start` (resolved config, output kind,
   latent layout + rgb factors), `status` (load/quantize/encode stages,
   from `print_and_status_update`), `progress` (step/total/eta), `latent`
   (raw fp16 tensor bytes + shape), `result` (file path/url + media meta),
   `error`, `end`. Text frames have empty payload.
3. **Latents stream raw, projection happens client-side.** The `start` frame
   carries `latent_layout` (`BCHW` / `BCFHW` / `audio`), the arch's
   `latent_rgb_factors` (+bias) and any scaling, so the browser projects
   latents to a canvas with a channels x 3 matmul per pixel and needs no
   per-arch code. Server options `stream.latents = "raw" | "none"`,
   `stream.every_n_steps`, and for video `stream.max_frames` /
   `stream.downsample` to cap frame size (a 30-step video run at full latent
   res is hundreds of MB; images are ~0.5 MB/step). Archs without known
   factors still stream latents; the client falls back to a per-channel
   grayscale/PCA preview. A server-side `preview="jpeg"` mode (full or
   TAE decode) is a later add, not v1.
4. **Endpoint discovery = ephemeral loopback port + a file in the job
   folder.** Engine binds `127.0.0.1:0`, writes
   `<training_folder>/<job_name>/engine.json` (`host, port, token, pid,
   started_at`) and logs it. No fixed port, nothing network-exposed, no
   collisions, works identically in docker (same container as the UI) and
   on Windows (uvicorn has no named-pipe support, so a loopback TCP port is
   the portable stand-in for "no port"). No DB schema change (a new Job
   column would crash the jobs list for anyone who has not run `update_db`).
   The UI already resolves `<training_folder>/<job.name>` for `plugin.html`.
   Requests carry the per-launch `token` (proxy adds it) so other local
   processes cannot drive the engine.
5. **One engine process per GPU set, one generation at a time, FIFO queue**
   (`GET /queue`, `POST /cancel`). Multiple engine jobs on different GPUs are
   independent; the UI picks by job.
6. **Hot-swap via a component pool hooked into the mixin funnel** (no
   rewrites of the 35 holder `load_model`s). See "Component pool".
7. **LoRA in v1 = part of the transformer's pool identity** (`lora_path`,
   `assistant_lora_path`, network multiplier apply through the existing
   holder paths; a different LoRA means the transformer entry reloads; TEs/
   VAEs stay shared). Hot-swappable unmerged LoRA on a resident transformer
   is v2.
8. **Server = FastAPI + uvicorn**, run on a thread inside the job process
   (generation stays on the main thread). Requires adding `fastapi` and
   `uvicorn` to `requirements.txt` (Jaret's call; fallback is a stdlib
   `ThreadingHTTPServer` with chunked responses — strictly worse, but zero
   deps).
9. **Model registry served by the engine.** `GET /models` returns every
   registered arch with capabilities + defaults (modality, control-image
   requirement, size divisibility, native res, default steps/CFG/frames/fps,
   latent layout, rgb factors). The table lives in python
   (`toolkit/models/registry.py`, seeded from `MODEL_TESTS` in
   `testing/test_model_loading.py` + per-holder class attrs) so the UI reads
   one source of truth at runtime instead of a second copy in `options.tsx`.

## Architecture

```
extensions_built_in/inference_engine/
  __init__.py            # InferenceEngineExtension (uid "InferenceEngine")
  InferenceEngine.py     # process: UI status/stop plumbing, server lifecycle
  server.py              # FastAPI app: /health /models /generate /cancel /queue /unload
  engine.py              # Engine: request queue, holder cache, pool orchestration
  protocol.py            # frame encoder (python) — mirrored by ui/src/utils/engineStream.ts
  latent_preview.py      # latent layout + rgb factor tables per arch family
toolkit/models/v2/pool.py       # ComponentPool (identity, residency, eviction)
toolkit/models/registry.py      # arch capabilities/defaults (shared with tests + UI)
toolkit/ui_job_status.py        # DB status/stop helpers lifted from BaseCaptioner
                                # (engine uses it; captioner can adopt later, optional)
testing/test_inference_engine.py
ui/src/app/api/inference/[...path]/route.ts   # proxy to the engine (streams)
ui/src/app/generate/page.tsx                   # Generate page
ui/src/components/generate/*                   # model picker, prompt form, preview canvas, gallery
ui/src/utils/engineStream.ts                     # frame parser + latent->canvas projection
```

### Process (`InferenceEngine.py`)

- Config:
  ```yaml
  job: extension
  config:
    name: inference_engine
    process:
      - type: InferenceEngine
        sqlite_db_path: ./aitk_db.db
        device: cuda
        engine:
          output_folder: <training_folder>/<name>/outputs   # results, served by /api/files
          keep_resident_gb: 0        # 0 = keep only what the last request needs
          idle_unload_seconds: 0     # 0 = never auto-unload
          default_model: {arch: zimage, name_or_path: Tongyi-MAI/Z-Image-Turbo}  # optional warm load
  ```
- `run()`: status "running / Starting server" -> start uvicorn on a thread
  (port 0) -> write `engine.json` -> optional warm load -> main thread loops
  on the request queue, executing generations; `info` reflects
  "Idle — zimage (Tongyi-MAI/Z-Image-Turbo)" / "Generating 12/30" / "Loading
  qwen_image: Quantizing transformer". Stop flag -> SIGINT (captioner
  pattern) -> cancel in-flight request, shut the server, delete
  `engine.json`, free the pool, status "stopped".
- Also runnable headless: `python run.py config/inference_engine.yaml`
  prints the endpoint; no UI needed (useful for tests).

### Engine (`engine.py`)

- `GenerateRequest`: `{model: ModelConfig kwargs (arch, name_or_path,
  extras_name_or_path, quantize, qtype, quantize_te, qtype_te, low_vram,
  layer_offloading..., lora_path, dtype), sample: GenerateImageConfig kwargs
  (prompt, negative_prompt, width, height, num_inference_steps,
  guidance_scale, seed, num_frames, fps, ctrl_img[_1..3] as server path or
  uploaded asset id, output_ext), stream: {latents, every_n_steps,
  max_frames}}`. Uploads: `POST /assets` stores control images under the job
  folder and returns an id/path.
- Holder handling: `get_model_class(model_config)`, build the holder with
  the pool active, `load_model()`, then `pool.release_unused()`. The holder
  is kept as "active" and reused verbatim for the next request when its
  ModelConfig is identical (fast path: no rebuild). A new ModelConfig
  rebuilds the holder; the pool makes the rebuild cost = only the changed
  components.
- Sampling: `sd.generate_images([gen])` with the step hook installed;
  `add_status_update_hook` feeds `status` frames; the `_after_sample_img`
  hook plus the saved file feed `result`. Output goes to
  `engine.output_folder` with a request-id filename; the frame carries the
  path and the `/api/files` url. Cancel = flag checked in the step hook,
  raising a `GenerationCancelled` that `generate_images` cleanup tolerates
  (it already restores device/rng state in its epilogue; verify the
  exception path leaves the holder usable).

### Step hook (`BaseModel`)

- `BaseModel.sample_step_hook: Optional[Callable]` and, inside
  `generate_images`, a context manager that wraps `pipeline.scheduler.step`
  to call `hook(step_index, num_steps, timestep, latents)` with the
  scheduler's returned sample. Zero cost when no hook is set. Custom loops
  that bypass the scheduler (found by the per-arch streaming test) get an
  explicit `self._emit_sample_step(...)` call. Legacy monolith: same wrap on
  its pipeline scheduler.
- Latent extraction from the DTO/carrier: video latents `BCFHW`, ltx/minimax
  AV extras (audio rows) ride the DTO — the `latent` frame sends the main
  tensor; audio extras optional.

### Component pool (`toolkit/models/v2/pool.py`)

- Identity key = `(component class, resolved source [repo/subfolder or file
  path after comfy resolution], dtype, qtype (incl. ARA suffix), offload
  fraction, final device, lora identity for transformers)`. Computed inside
  `OstrisModelMixin.load` from exactly the args it already receives, so a
  holder's `component_load_kwargs(role)` output determines sharing (a TE
  loaded quantized convrot8 on cuda:0 is one entry; the same TE bf16 on cpu
  is another).
- Wiring: a process-global `ComponentPool.current` context; `load` checks
  it first (hit = return resident module, skip source+post_load), registers
  on miss. Legacy `adopt_component` funnel registers too (consume side stays
  full reload for legacy archs). Tokenizers/processors pooled by repo.
- Residency ops: `mark_used(key)` per request; `release_unused(keep)` =
  MemoryManager.detach/free, drop refs, `gc.collect()`,
  `torch.cuda.empty_cache()`; `bytes(key)` for accounting;
  `keep_resident_gb` LRU keep-warm; eviction before a load when
  `torch.cuda.mem_get_info` says the incoming component will not fit.
- Safety: entries must not be mutated by holders in ways that change their
  identity. Audit as part of step 3: holders that merge LoRAs into weights,
  `.to(device)` moves under low_vram (device is in the key — a moved module
  must be moved back or re-keyed on release), `MemoryManager.attach` (offload
  is in the key), FakeTextEncoder swaps (`toolkit/unloader.py`), per-arch
  patches applied in `load_model` after `load` (e.g. vision-tower drops:
  fine, they are class-level config and thus part of the class identity).
- Counters exposed on `/health`: hits/misses/evictions, resident entries with
  bytes and last-used — the tests assert on these.

### Server (`server.py`)

- `GET /health` `{ok, busy, queue, active: {arch, name_or_path, config},
  pool: [...], vram: {used, total}}`; `GET /models`; `POST /generate`
  (stream); `POST /cancel/{id}`; `GET /queue`; `POST /unload`
  (`{all: true}` or component keys); `POST /assets` (multipart upload);
  `GET /outputs` (list results). Token check on everything except `/health`.

### UI

- Proxy: `ui/src/app/api/inference/[...path]/route.ts` — finds the running
  `job_type = "inference"` job (query param `job` to pick one when several
  GPUs run engines), reads its `engine.json` (cached a few seconds),
  forwards method/headers/body with the engine token, returns
  `new Response(upstream.body)` so chunked frames flow through untouched.
  `export const dynamic = 'force-dynamic'`, no body size limit issues
  (control images go through `/assets`, multipart).
- `/generate` page (sidebar "Generate"): engine strip (pick/start/stop the
  engine job on a GPU — creates a job via `/api/jobs` with
  `job_type: 'inference'` and the config above, then start + startQueue,
  exactly like `CaptionDatasetModal.saveJob`; shows health), model panel
  (arch select from `/models`, name_or_path creatable select with the arch's
  defaults, quantize/qtype/low_vram/LoRA), prompt panel (prompt/negative,
  size with divisibility snapping, steps/CFG/seed/frames/fps, control image
  upload), preview panel (canvas fed by the stream, step/eta, cancel),
  results gallery (files from the engine output folder via `/api/files`,
  click to open; video/audio players).
- Jobs list / action bar: `job_type === 'inference'` rows show "Inference
  Engine" with a link to `/generate`; stop only (no edit/restart-from-step
  UI). Dashboard active-job widget same treatment as caption.
- `ui/src/utils/engineStream.ts`: async frame reader over `fetch` +
  `ReadableStream`, event emitter, `projectLatentToImageData(latent, layout,
  rgbFactors)`; video: shows a chosen frame (scrubber) — audio: heatmap.

### Tests (`testing/test_inference_engine.py`)

Mirrors `test_model_loading.py` conventions (`--arch`, `--all` = subprocess
per arch, `--allow-download`, HF offline default, SKIP classification,
outputs under `testing/.engine_test_outputs/<arch>`), reusing its
`MODEL_TESTS` entries for per-arch model/sample settings:

1. `--arch X`: start the engine in-process on an ephemeral port, POST
   `/generate`, consume the stream, assert `start` -> >=1 `progress` ->
   >=1 `latent` (shape/dtype/layout sanity, finite values) -> `result`
   (file exists, > 1 KB, right extension for the modality) -> `end`;
   record timings + VRAM like the loading test.
2. `--swap A,B[,C]`: sequential requests; assert pool hits for shared
   components (e.g. zimage -> zimage_l2p shares Qwen3 TE + KLVAE; wan21 ->
   wan22_5b shares UMT5; qwen_image -> qwen_image_edit shares TE + VAE),
   assert VRAM after B does not include A's DiT, assert a second A request
   is a holder fast path (no loads).
3. `--cancel`: cancel mid-run, assert the stream ends with `error/cancelled`
   and the next request on the same holder succeeds.
4. `--all`: (1) for every registry arch, then a summary + `report.md`.

## Steps

Each step lands runnable + tested before the next.

- [x] **1. Skeleton job + server** (2026-09-10) — `InferenceEngineExtension`
      (uid `InferenceEngine`), process with the sqlite status/stop plumbing
      lifted into `toolkit/ui_job_status.py`, FastAPI + uvicorn on
      `127.0.0.1:0` (`server.serve_in_thread`), `engine.json` in the job
      folder, `/health` `/models` `/generate` `/cancel` `/queue` `/unload`
      `/assets` `/outputs`, headless yaml. `fastapi`/`uvicorn` pinned in
      requirements_base.txt.
- [x] **2. Streaming** — binary frame protocol (`protocol.py`, mirrored by
      `ui/src/utils/engineStream.ts`), `BaseModel.sample_step_hook` via a
      scheduler.step wrap in `generate_images` (+ `_emit_sample_step` for
      hand-rolled loops; ace_step wired), latent layouts BCHW/BCFHW/BLC with
      flux-style packed-token unpacking and video frame subsampling, per-arch
      preview tables (`latent_preview.py`, numbers from ComfyUI's
      latent_formats), cancel (step hook raises; holder verified reusable:
      the rerun after a cancel reproduced the same file byte-for-byte).
      Verified: zimage — 8 progress + 8 latent frames (1x16x64x64, Flux
      preview) + result; `--cancel` PASS.
- [x] **3. Component pool + hot-swap** — `toolkit/models/v2/pool.py`;
      `OstrisModelMixin.load_model` consults/registers the active pool by
      source key (class, name_or_path, dtype, config, subfolder, comfy flag,
      kwargs); `aitk_post_load` is idempotent for an identical policy and
      detaches offloading before re-applying a different one. Engine: drop
      holder → begin_request → build → release_unused; CUDA OOM during a
      build frees untouched entries and retries once. Verified:
      zimage→zimage_l2p→zimage PASS (TE shared on the way back; l2p's
      transformer/VAE bypass the mixin so they reload — see step 4);
      wan21→wan22_5b: UMT5 TE shared (pool hit), wan21 DiT+VAE released.
- [ ] **4. Coverage** — engine test across the registry. PASS so far
      (2026-09-10, 5090 with ~16 GB free next to a training job, convrot8):
      zimage (image, Flux preview), wan21 + wan22_5b (video BCFHW, animated
      webp), ace_step_15 (audio BLC, mp3), sd1 (legacy monolith: the hook
      is installed in its own generate_images via toolkit/sample_step_hook),
      zimage `--cancel`. Known gaps: holders whose component loads
      bypass `load_model` (zimage_l2p's from_pretrained + pixel conversion,
      legacy monolith adopt path) do not pool; omnigen2 builds its scheduler
      inside get_generation_pipeline (hook still applies — wrapping happens
      after); custom loops without scheduler.step need `_emit_sample_step`
      as they are found. Full `--all` sweep not run (shared GPU); run it
      when a free card is available.
- [x] **5. UI** — `/api/inference/[...path]` streaming proxy
      (`ui/src/server/inferenceEngine.ts` resolves the running inference
      job's engine.json, checks the pid, injects the token), `/generate`
      page (engine start/stop per GPU, model + prompt forms fed by
      `/models`, live latent canvas with frame scrubber, results gallery via
      `/api/files`), sidebar entry, inference rows in the queue table /
      dashboard widget / job page, `startJob.ts` injects
      `engine.job_folder`. Type-checked and built in an isolated copy of the
      UI (the live `next start` on this box was not rebuilt — run
      `npm run build` to pick it up, the worker change needs it too).
      Proxy verified end to end against that isolated build with a copy of
      the db: status → health → models → streamed generate (first frame
      arrives immediately, progress frames live, result at the end) → stop
      via the Job.stop flag (engine exits 0, row 'stopped', engine.json
      removed, status reports no engine).
- [ ] **6. Hardening** — done: FIFO queue + `/queue`, OOM-retry eviction,
      stale engine.json ignored when its pid is dead, `wait=1` JSON mode.
      Open: idle unload, `keep_resident_gb` LRU keep-warm, pre-load VRAM-fit
      eviction (today untouched entries are only freed after the build or
      on OOM, so a tight card pays a reload instead of sharing when the
      DiT loads first), Windows relay launch check, docker compose smoke,
      pooling for the custom-path holders.

## Open questions

1. Chunked HTTP stream vs WebSocket (decision 1).
2. Raw latents + client-side projection vs server-rendered previews
   (decision 3). Both can coexist later; which is v1?
3. `engine.json` in the job folder vs a new Job DB column (decision 4).
4. OK to add `fastapi` + `uvicorn` to `requirements.txt`?
5. LoRA as transformer identity in v1 (decision 7) — acceptable?
6. Should the per-arch registry (`toolkit/models/registry.py`) also become
   the source for `ui/src/app/jobs/new/options.tsx` in the same work, or
   stay engine-only for now?
7. Does the engine also need a non-streamed `POST /generate?wait=1` JSON
   mode for external scripts (cheap to add alongside)?
