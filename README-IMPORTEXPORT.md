# ai-toolkit — Fork Changelog

This document describes all additions and modifications made to this fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit).

---

## Features

### Export Job

Adds an **Export Job** option to the gear menu on each training job card.

Clicking it triggers a browser download of a `.zip` archive containing everything needed to resume the job on another machine:

- `manifest.json` — job metadata (name, config, step count, dataset paths)
- `output/` — all checkpoint files (`*.safetensors`) and `config.yaml`
- `datasets/` — all dataset folders referenced by the job config, including latent caches

The ZIP is streamed directly to the browser with no intermediate file written to disk. Only available for `job_type: train` jobs.

**New file:** `ui/src/app/api/jobs/[jobID]/export/route.ts`

---

### Import Job

Adds an **Import Job** button next to "New Training Job" in the jobs page header.

Clicking it opens a modal where the user can drag-and-drop (or click to select) a `.zip` file previously exported from another machine. The import:

1. Streams the upload to a temp file (no full-file buffering in RAM)
2. Reads `manifest.json` from the ZIP
3. Extracts the output folder and all dataset folders to the correct paths on the current machine
4. Rewrites all absolute paths in the job config (`training_folder`, `sqlite_db_path`, dataset `folder_path`) to match the current machine
5. Renames checkpoint files if the job is imported with a different name
6. Creates a new DB record with the correct step count and `stopped` status

After a successful import, the jobs table refreshes automatically.

**New files:**
- `ui/src/app/api/jobs/import/route.ts`
- `ui/src/components/ImportJobModal.tsx`

**Modified files:**
- `ui/src/app/jobs/page.tsx` — Import Job button and modal wiring
- `ui/src/components/JobActionBar.tsx` — Export Job menu item

---

### Checkpoint Detection Bug Fix

`BaseSDTrainProcess.py` previously used `os.path.getctime` to find the latest checkpoint file. On Linux, `ctime` is the inode change time and resets when files are copied — which means after importing a job, the wrong checkpoint could be loaded.

Fixed by sorting checkpoint files by the step number embedded in the filename (`{name}_{step:09d}.safetensors`) instead of by filesystem timestamps.

**Modified file:** `jobs/process/BaseSDTrainProcess.py`

---

### next.config.ts

- Added `unzipper` to `serverExternalPackages` (required for ZIP extraction in the import route)
- Removed deprecated `devIndicators.buildActivity` (removed in Next.js 15.5+)
- Added `experimental.serverActions.bodySizeLimit: '100gb'` and `experimental.middlewareClientMaxBodySize: '100gb'` to support large ZIP uploads

**Modified file:** `ui/next.config.ts`

---

## npm Dependencies Added

```
unzipper
@types/unzipper
busboy
@types/busboy
```

---

## Installing on Another ai-toolkit Instance

A `Makefile` is included to deploy this fork's changes onto a separate ai-toolkit installation without manually copying files.

```bash
# First install — backs up the files that will be modified
make install DEST=/path/to/other/ai-toolkit

# Update after a git pull on the destination (reverses old patch, re-applies new one)
make update DEST=/path/to/other/ai-toolkit

# Revert all changes and restore original files
make uninstall DEST=/path/to/other/ai-toolkit
```

**How patching works:**

The three files that existed upstream and were modified (`BaseSDTrainProcess.py`, `jobs/page.tsx`, `JobActionBar.tsx`) are deployed as **patch files** (`patches/`) rather than full file copies. This means:

- If the destination has a newer upstream version of those files, the patch is applied on top of it — upstream changes are preserved
- If the patch context is incompatible (the destination has changes in the same lines as our patch), the command stops and reports the conflict with instructions to resolve it manually
- The three new files (`export/route.ts`, `import/route.ts`, `ImportJobModal.tsx`) are always copied directly since they did not exist upstream

`make install` also creates `.aitk-bak` backup copies of the original files before patching, used as a fallback by `make uninstall`.

---

## Files Summary

| File | Status | Description |
|---|---|---|
| `jobs/process/BaseSDTrainProcess.py` | Modified | Checkpoint sort fix |
| `ui/next.config.ts` | Modified | unzipper extern, body size limits |
| `ui/src/app/jobs/page.tsx` | Modified | Import Job button + modal |
| `ui/src/components/JobActionBar.tsx` | Modified | Export Job menu item |
| `ui/src/app/api/jobs/[jobID]/export/route.ts` | New | Export ZIP endpoint |
| `ui/src/app/api/jobs/import/route.ts` | New | Import ZIP endpoint |
| `ui/src/components/ImportJobModal.tsx` | New | Import modal component |
| `Makefile` | New | Install/update/uninstall helper |
| `patches/BaseSDTrainProcess.patch` | New | Patch for BaseSDTrainProcess.py |
| `patches/jobs_page.patch` | New | Patch for jobs/page.tsx |
| `patches/JobActionBar.patch` | New | Patch for JobActionBar.tsx |
| `patches/next_config.patch` | New | Patch for next.config.ts |
