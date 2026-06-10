# Claude Code skills for the AI Toolkit training workflow

These are [Claude Code](https://claude.com/claude-code) skills that drive the
end-to-end LoRA training workflow in this repo — from config generation
through captioning, remote-GPU training on RunPod, and checkpoint review.
Each `<name>/SKILL.md` is a self-contained skill; some carry a `references/`
dir with deeper material.

## Activating them

Claude Code auto-discovers skills from `.claude/skills/` (which is
gitignored), not from this tracked `skills/` dir. To make these active in a
clone, symlink (recommended — edits stay in sync with git) or copy each one:

```bash
mkdir -p .claude/skills
for s in skills/*/; do
  name=$(basename "$s")
  ln -sfn "../../skills/$name" ".claude/skills/$name"
done
```

Then start (or restart) Claude Code in the repo and the skills trigger by
name or by their `description` triggers.

## The workflow

| Stage | Skill | Use when |
|---|---|---|
| Config | `ai-toolkit-lora-config` | "train a LoRA on X" — generates the training YAML |
| Config (Klein) | `flux2-klein-lora-config`* | Flux.2 Klein-specific config recipe |
| Captioning | `ai-toolkit-gemini-captioner` | generate per-dataset Gemini captions |
| Caption QA | `style-vs-content-caption-auditor` | audit captions for leakage before training |
| Dataset triage | `ai-toolkit-dataset-diagnostics` | "no images found", crashes before step 0, stale cache |
| DOP tuning | `dop-class-advisor`* | pick `diff_output_preservation_class` |
| **Remote launch** | `ai-toolkit-remote-launch` | "train this on RunPod" — preflight + provision + sync + launch |
| **Remote monitor** | `ai-toolkit-remote-monitor` | "check on my run" — watch loop, pull, drive review |
| **Remote teardown** | `ai-toolkit-remote-teardown` | "tear it down" / "is anything still billing" |
| Review | `ai-toolkit-sample-reviewer` | "review my samples / pick a checkpoint" |

\* `flux2-klein-lora-config` and `dop-class-advisor` are model-/parameter-
specific helpers; the rest form the universal path.

The three **remote** skills wrap `scripts/remote/cli.py` (the hosted-GPU
pipeline — see `scripts/remote/README.md`). The remaining skills are
model-training methodology and run locally regardless of where training
executes.

## Typical end-to-end (RunPod)

```
ai-toolkit-lora-config            # generate config from reference images
ai-toolkit-gemini-captioner       # caption the dataset
style-vs-content-caption-auditor  # audit captions
ai-toolkit-remote-launch          # preflight -> up (provision/sync/launch)
ai-toolkit-remote-monitor         # watch --once --json loop + sample review
ai-toolkit-sample-reviewer        # pick the checkpoint
ai-toolkit-remote-teardown        # down / rescue / confirm nothing billing
```

Requires `RUNPOD_API_KEY`, `RUNPOD_STOP_API_KEY`, and `HF_TOKEN` in `.env`
(see `scripts/remote/README.md` §1).
