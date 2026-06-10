# AI Toolkit Output Layout

Everything you need to know about navigating a training run's output folder.

## Directory structure

For a config with `training_folder: "output"` and `config.name: "my_run"`:

```
output/my_run/
├── samples/
│   ├── 1730812345678__000000250_000.jpg
│   ├── 1730812345678__000000250_001.jpg
│   ├── ...
│   ├── <gen_time_ms>__<step>_<count>.jpg
│   └── ...
├── my_run.safetensors                  # final checkpoint
├── my_run_000000250.safetensors        # interval checkpoint at step 250
├── my_run_000000500.safetensors
├── ...
├── my_run.yaml                         # snapshot of the training config
└── optimizer.pt                        # optimizer state (ignore)
```

The `training_folder` may be overridden in the YAML — read it from the config rather than assuming `output/`.

## Sample filename format

```
<gen_time_ms>__<step_zero_padded_9_digits>_<count_zero_padded_3_digits>.<ext>
```

Source: `jobs/process/BaseSDTrainProcess.py` builds `[time]_<step>_[count].<ext>` and `toolkit/config_modules.py` does the replacement.

Examples:
- `1730812345678__000000250_000.jpg` — step 250, prompt index 0
- `1730812345678__000001500_007.jpg` — step 1500, prompt index 7
- `1730812345678__000003000_011.jpg` — step 3000, prompt index 11

### Parsing the filename

```python
import re
from pathlib import Path

PATTERN = re.compile(r"^\d+__(\d{9})_(\d+)\.\w+$")

def parse_sample_name(path: Path):
    m = PATTERN.match(path.name)
    if not m:
        return None
    step = int(m.group(1))
    count = int(m.group(2))
    return step, count
```

### Bash one-liner to group samples by step

```bash
ls output/<run>/samples/*.jpg | awk -F'__|_' '{print $2}' | sort -u
```

That prints every unique step that has samples.

## Mapping `count` back to a prompt

The `count` index is the position of the prompt in `sample.prompts` from the YAML. To map a sample back to what it was supposed to render:

```python
import yaml
config = yaml.safe_load(open("config/.../my_config.yaml"))
prompts = config["config"]["process"][0]["sample"]["prompts"]
# count 7 → prompts[7]
```

Note: ai-toolkit substitutes `[trigger]` with the `trigger_word` from the config, but the YAML still shows `[trigger]`. Remember that the *rendered* prompt for `[trigger] a red bicycle` actually went to the model as `<actual-trigger> a red bicycle`.

For `samples:` (list-of-dicts form with `prompt:` and optional `ctrl_img:`), the index still maps the same way.

## Identifying control prompts

Control prompts (the diagnostic for "is the LoRA bleeding into base behavior?") are the ones in `sample.prompts` that do **not** contain `[trigger]` or the trigger word itself. Scan each prompt:

```python
trigger = config["config"]["process"][0]["trigger_word"]
controls = [i for i, p in enumerate(prompts)
            if "[trigger]" not in p and trigger not in p]
```

These indices are the most important to look at — they expose over-baking.

## Listing checkpoints

```bash
ls output/<run>/<run>_*.safetensors | sort
```

The step number in the checkpoint filename is also zero-padded to 9 digits:
- `my_run_000000250.safetensors` = checkpoint at step 250

This matches the step number in the sample filenames, so checkpoint X corresponds to the samples generated at the same step X.

The "final" checkpoint without a step number (`my_run.safetensors`) corresponds to the last training step.

## Inferring the run folder from a config

The output run folder is `<training_folder>/<config.name>`. Both come from the YAML:

```yaml
config:
  name: "my_run"           # <-- this is the run name
  process:
    - type: 'sd_trainer'
      training_folder: "output"   # <-- and this
```

If the user gives you a config path, you can compute the output folder from these two fields.

## Dataset folder

Pulled from `config.datasets[0].folder_path` in the YAML. Multiple datasets are possible (e.g. for combined character + style); iterate over the list.

## Merging two checkpoints

ai-toolkit ships `scripts/merge_loras.py` which concatenates ranks (the mathematically clean way to merge LoRAs without SVD recompression).

Usage:

```bash
python scripts/merge_loras.py \
  --lora_a output/<run>/<run>_<step_A>.safetensors --weight_a 1.0 \
  --lora_b output/<run>/<run>_<step_B>.safetensors --weight_b 0.5 \
  --output output/<run>/merged_<A>_<B>.safetensors
```

Notes:
- `weight_a` / `weight_b` scale each contribution. For "use both equally" use `1.0 / 1.0`. For "mostly A, dab of B" use `1.0 / 0.3`.
- The output file's rank = `rank_A + rank_B`, so the file roughly doubles in size.
- Both LoRAs MUST share the same base model and target the same modules. Two checkpoints from the same training run always qualify.
- Trigger word and inference settings stay the same as the source LoRAs.

### When to suggest a merge

Suggest only when the two checkpoints have *genuinely complementary* strengths the user can't pick a single winner over. Most common cases:

1. **Style + structure**: an early-style checkpoint nails the texture/palette but lacks subject coherence; a late checkpoint has the right detail but lost the texture. Merge at `1.0 / 1.0`.
2. **Triggered + clean controls**: a late checkpoint has the best triggered outputs but bleeds on controls; an earlier checkpoint is cleaner on controls. Merge late at `1.0` + early at `0.3-0.5` to dial back the bleed.

Do not suggest merges to combine "two similarly-good checkpoints with minor differences" — the rank cost and slight quality loss aren't worth it.
