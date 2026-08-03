# AI Toolkit Manager

Self-contained install/update manager for this checkout of AI Toolkit. Runs
with any Python >= 3.8 and **no dependencies**, so it works before the
training environment exists.

```bash
python3 -m manager install     # first-time setup: venv + torch + requirements
python3 -m manager check       # is an update available / are deps out of sync?
python3 -m manager update      # git pull, then sync deps + run migrations
python3 -m manager launch      # start the web UI (http://localhost:8675)
python3 -m manager doctor      # diagnose problems
```

## Design

- **The install logic lives in the repo it installs.** Every commit knows how
  to install itself; external frontends (the desktop launcher, `install.sh`,
  `install.ps1`) just shell out to this CLI and stay dumb. Machine-readable
  output via `check --json` / `detect --json`.
- **Hardware → spec mapping** is in [spec.py](spec.py). One universal torch
  pin (2.13.0 / torchvision 0.28.0 / torchaudio 2.11.0) on every platform:
  cu130 wheels when the driver supports CUDA 13 (cu126 fallback for older
  drivers, refused outright on Blackwell GPUs which need cu130), same stack +
  Python 3.11 + `dgx_requirements.txt` on DGX/Grace, PyPI wheels on Mac,
  rocm7.1 (experimental) for AMD, `--cpu` to force a CPU install. **Torch
  pins there must be updated together with the README install instructions,
  run_mac.zsh, and dgx_instructions.md.**
- **Accelerators everywhere wheels exist**, via per-spec `extra_packages`
  (installed after requirements with `--upgrade` so they override pins) and
  `optional_packages` (installed one-by-one, warn-only on failure):
  `torchcodec==0.15.0` on all platforms; flash-attn 2.8.3 prebuilt wheels
  (mjun0812) on Linux x86_64/aarch64 + Windows; NATTEN 0.21.7 wheels
  (whl.natten.org) on Linux both arches; triton bundled with torch on Linux
  and `triton-windows` 3.7.x on Windows. No flash-attn/NATTEN/triton on Mac,
  no NATTEN on Windows (no wheels exist).
- **The torch stack is pinned against the resolver.** torch is an unpinned
  transitive dep of timm/peft/accelerate/torchvision, so anything in
  `requirements*.txt` that conflicts with what the pinned torch needs makes the
  resolver silently backtrack to an older torch rather than fail — and prebuilt
  accelerator wheels then reinstall a plain PyPI torch over the GPU one,
  leaving torchvision/torchaudio's C++ extensions linked against a libtorch
  that is gone. Every install pass after torch therefore carries a generated
  constraints file (`torch==X+cu130`, ...) plus per-package `--find-links` to
  the pytorch index, and `_verify_torch` re-checks (and repairs) the trio
  before the optional accelerators are import-tested and again at the end.
  Requirements files must never pin a torch dependency below what torch needs
  (torch 2.13 wants `setuptools>=77.0.3`).
- **`ui/package-lock.json` is never modified by installing.** A plain
  `npm install` re-derives the lockfile for whichever machine runs it — on
  Windows it strips the `libc` fields off the Linux-only optional binaries
  (`@next/swc-linux-*`, rollup, lightningcss), on Linux it adds them back — so
  with the install baked into `npm run build_and_start` every user got a dirty
  tree on every launch, which then blocks `manager update` (a dirty tree aborts
  the pull). `nodejs.ensure_ui_deps` owns the install instead: `npm install
  --no-save`, gated on a hash of `ui/package.json` + `ui/package-lock.json`
  (stored in the venv state), with the lockfile bytes snapshotted and restored
  either way. `manager launch` therefore runs `npm run db_build_start`, not
  `build_and_start`; the latter still exists for the manual
  `cd ui && npm run build_and_start` flow in the README and calls the same
  non-writing install via `npm run install_deps`.
- **Nothing global is ever installed.** FFmpeg (shared builds — the libs
  torchcodec dlopens) goes to `.ffmpeg/` ([ffmpeg.py](ffmpeg.py)), Node
  (when the system lacks >= 20) to `.node/` ([nodejs.py](nodejs.py)), the uv
  binary (when absent) to `.uv/` ([uvbin.py](uvbin.py)) with uv-managed
  Pythons kept in `.uv/python/` via `UV_PYTHON_INSTALL_DIR`, and on Windows
  without git, portable MinGit to `.mingit/` ([gitwin.py](gitwin.py)) — all
  inside the repo and gitignored. The first clone on a git-less Windows
  box is handled by the bootstrap layer (install.ps1 / desktop launcher),
  which downloads MinGit itself and moves it into the checkout afterwards. `manager launch` puts them on PATH (and
  LD_LIBRARY_PATH on Linux) for the whole UI/training process tree, and a
  generated `sitecustomize.py` in the venv exposes ffmpeg to any direct use
  of the venv python (plus `os.add_dll_directory` on Windows).
- **Hostile-environment hardening** (learned from the community Windows
  installer): every python/pip subprocess runs with PYTHONPATH/PYTHONHOME/
  CONDA/PYENV/PIP_* scrubbed from the env; git runs with
  `GIT_LFS_SKIP_SMUDGE=1`; git-pinned requirements (diffusers) are
  force-reinstalled when requirements change since pip skips unchanged
  version numbers; `launch` polls the UI port and opens the browser when
  ready (`--no-browser` to disable, auto-skipped on headless boxes).
- **uv is used when present** (fast installs, auto-downloads the right
  Python); plain `venv` + `pip` otherwise. The venv is created at `.venv/`
  (an existing `venv/` is also respected, matching `ui/cron/pythonPath.ts`).
- **State** (requirements hash, applied migrations) lives inside the venv
  (`aitk_manager_state.json`) — deleting the venv resets everything.
- **Update flow**: `update` pulls fast-forward only, then **re-execs**
  `python -m manager sync` so the freshly pulled manager code — not the stale
  in-memory copy — performs its own dependency sync and migrations.
  **Local work is never overwritten**: a dirty tree aborts the update by
  default (untracked files don't count), `--auto` (used by the run_* scripts)
  warns and skips the pull instead so launching still works, and there is no
  reset/clean anywhere — even `--force` relies on git itself refusing to
  clobber modified files.
- **Migrations** ([migrations.py](migrations.py)): one-time post-update steps,
  each applied at most once per environment.
