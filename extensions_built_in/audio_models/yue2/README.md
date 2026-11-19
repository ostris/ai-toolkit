# YuE2 (m-a-p/YuE2-3B, text-to-music)

ai-toolkit integration for YuE2: an AR expert that writes an ABC lead sheet and then the song's codec
tokens, and a NAR expert that renders those tokens into VAE latents. Training conditions the AR on codec
tokens read from real songs by an audio -> semantic-token tokenizer. The official one is unreleased, so the
tokenizer is a model you supply (see below).

The base trainer is [yue2_model.py](yue2_model.py); the training additions documented here live in the
subclass in [yue2_local.py](yue2_local.py), which `__init__.py` registers under the same `yue2` arch.

## Semantic tokenizer

### WannabeTokenizer (recommended)

[`ManyOtherFunctions/Yue2-WannabeTokenizer`](https://huggingface.co/ManyOtherFunctions/Yue2-WannabeTokenizer)
is a tokenizer in the YuE2 report's form: a causal MERT-v2 encoder adapted to YuE2's ids plus a cosine
codebook quantizer. Each file is unified, with the encoder and the head inside one `.safetensors`.

1. Download `WannabeTokenizerV3.safetensors` from that repo.
2. Copy it into the models folder: `<MODELS_PATH>/ai_toolkit/` (`models/ai_toolkit/` in the repo unless
   `MODELS_PATH` is set).
3. Point the job at it by file name:

```yaml
model:
  arch: "yue2"
  model_kwargs:
    semantic_head_path: "WannabeTokenizerV3.safetensors"   # a bare name = the file in <MODELS_PATH>/ai_toolkit/
```

Unlike the Mothersuperior community tokenizer, it needs **no NAR companion module**: leave
`merge_nar_lora` off (the default), and you need no MERT-v2-FullSong folder either, because the file
carries its own encoder. The file also names the loudness band it was trained to read songs at
(`head.level_band` in its metadata, used by `level_fit` below). Copy the file in before the first run,
because the band is read when the latent cache key is built, and a file that is not on disk yet falls
back to the default band.

`semantic_head_path` also takes a full path, an `org/repo/file` id (downloaded into the same folder), or
a training run's `runs/<run>/best/head.safetensors` with its encoder (`model.safetensors` +
`config.json`) beside it. The tokenizer's format is read from the file's `head.kind` metadata, and each
tokenizer gets its own latent cache.

### Mothersuperior community head

The default (`Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`) is a classifier over stock
MERT-v2-FullSong features, released together with a NAR LoRA that is meant to be merged with it
(`merge_nar_lora: true`). Its encoder is a local folder (see below).

## Local folders (never downloaded at run time)

Code that gets executed, and the encoder of the community head, are no longer pulled through
`trust_remote_code` or the Hugging Face cache. They are local folders you place once:

| `model_kwargs` key | Default | What goes in it | Needed for |
|---|---|---|---|
| `sheetsage_code` | `<MODELS_PATH>/ai_toolkit/SheetSage2` | `config.json` + `*.py` of `m-a-p/SheetSage2` at revision `55bfe14e32d8b663629b3a86b0f3285c2ca5da0b` (no weights) | `cot: full` / `melody`, unless every track has a manual `.abc` sheet (with either sheet writer: Refined's sheets are written by SheetSage2's exporter) |
| `mert_path` | `<MODELS_PATH>/ai_toolkit/MERT-v2-FullSong` | `config.json` + `model.safetensors` of `m-a-p/MERT-v2-FullSong` | the community head only |

SheetSage2's weights still come from `sheetsage_path`
(`Comfy-Org/YuE2/audio_encoders/sheetsage2_bf16.safetensors`) and download like the checkpoint. A missing
folder stops the run with a message that says what to put where.

## Sheets: SheetSage2 or SheetSage2-Refined

For `cot: full` / `melody`, a track without a manual `.abc` gets its sheet from a transcriber at cache time,
picked by `model_kwargs.sheet_transcriber`:

- `sheetsage2` (default): the released SheetSage2, the transcriber ComfyUI uses for cover mode.
- `refined`: [SheetSage2-Refined](https://github.com/anime-song/SheetSage2-Refined) by **anime-song**, a
  fine-tuned SheetSage2 with its own decoder plus rescoring heads for downbeats, meters, chords and keys. It finds
  bars and odd meters much more reliably than the original. Its code (MIT, 0.5.2) is vendored in
  [sheetsage2_refined/](sheetsage2_refined/). Its event stream goes through SheetSage2's own exporter, so the
  sheets stay in the ABC dialect the AR was trained on. Refined writes the vocal line in the octave actually sung,
  one below SheetSage2's and YuE2's own sheets, so the line is moved up an octave.

Refined's weights (CC BY-NC 4.0) live in a model folder, `model_kwargs.refined_path`. Unset, it is
`<MODELS_PATH>/ai_toolkit/SheetSage2-Refined`, and the files transcription needs (`config.json`, the decoder,
the bar features and the rescoring heads, about 250 MB) download there from `anime-song/SheetSage2-Refined` at
the pinned 0.5.2 revision. Its encoder is SheetSage2's own. It is rebuilt once from the SheetSage2 file you
already have, instead of downloading the 2.7 GB repack. A folder you name with `refined_path` must already be
complete. Refined sheets get their own latent cache, so switching transcribers never mixes the two.

Huge thanks to anime-song for SheetSage2-Refined, and for releasing its code under MIT.

## Dataset

```
my_songs/
  song.flac     audio
  song.txt      caption: tags, then a [Lyrics] line and the lyrics
  song.abc      optional: this track's ABC sheet, used instead of SheetSage2's transcription
```

- **Manual sheets**: `song.abc` beside the audio is the training sheet for that track. It has to match
  `cot` (no chords in a `melody` run). Adding or editing one rebuilds only that file's cache. Not
  supported with `do_separation`, whose stems need sheets of their own.
- **Pairing check**: YuE2 datasets stop before caching when a track's caption is empty (no `.txt`, a
  blank one, no `default_caption`) or when a `.txt` / `.abc` has no audio file beside it. Set
  `strict_pairing: false` on a dataset to train anyway. Other models can opt in with
  `strict_pairing: true`.

### Audio formats

wav, flac, mp3, ogg, aac and opus. Audio is decoded with PyAV ([toolkit/audio/decode.py](../../../toolkit/audio/decode.py)),
not `torchaudio.load`: the torchcodec backend returns Microsoft ADPCM WAVs at half or a quarter of their
length, still labelled with the original sample rate, so they train as sped-up audio with no error
(ostris/ai-toolkit#1063). The same decoder is used by the dataloader, the ACE-Step and Qwen2.5-Omni
captioners, ACE-Step latents, MiniMax-H3 reference audio and MelBandRoformer.

**m4a is a cursed format**, and this does not add it anywhere. An m4a whose edit list is missing or wrong
comes back shifted (21.33 ms in one measured case), and the tokenizer's 40-ms frame grid moves with it:
2 ms of shift changed about 20 % of a song's ids. Convert m4a to flac before training.

## Training options (`model_kwargs`)

| Key | Default | What it does |
|---|---|---|
| `semantic_head_path` | Mothersuperior v4 head | The tokenizer (see above). |
| `level_fit` | `band` | Loudness the tokenizer reads songs at. `band` (or `true`): a song outside the tokenizer's band (`head.level_band`, else -21 to -13 dBFS) moves to its nearest edge. `all`: every song moves to -16.5 dBFS. `false`: as recorded. Applies to the latents, the codec tokens and `[Reconstruct]` samples alike. |
| `phase_tta` | `false` | `true` reads each song at -10/-5/0/5/10 ms sub-frame shifts and averages the head's softmax before the argmax (or give a list of ms within +-20). Steadier ids, but by ear the singer's timbre suffers. |
| `train_experts` | `both` | `both` = one joint LoRA. `ar` = the planner only (the NAR is skipped; needs `ar_loss_weight` > 0). `nar` = the decoder only (AR loss off). A single-expert LoRA holds only that expert's keys. |
| `other_expert_lora_path` | none | With `train_experts` `ar` or `nar`: a finished LoRA held frozen on the other expert for the whole run. Its checkpoints carry that half too, so each one is a complete joint LoRA. |
| `base_lora_path` | none | A finished LoRA held frozen on the trained expert(s), with the new LoRA stacked on top, for staged runs. Checkpoints hold only the new delta and are stamped with their base; `merged_exports/` beside them gets each step as one complete LoRA (rank-concatenated, exact). |
| `ar_loss_on` | `all` | Which AR targets the next-token loss counts: `all`, `sheet` (the ABC sheet only; `abc_dropout` is then ignored) or `tokens` (the music only). The KL leash, when on, still covers every position. |
| `replay_dirs` | none | Exact-token replay: corpora in the minted layout (`<root>/tracks/<id>/semantic.npy` + `request.json` [+ `score.abc`]) whose codes the AR wrote itself. With probability `replay_ratio` (default 0.2 when dirs are given) a step adds one replay track's next-token loss, weighted `replay_weight` (default 1.0), so the AR keeps its own codes while it learns the new songs. Logged as `loss/ar_replay_ce`. |
| `sample_cfg_negative` | `text` | What the semantic CFG negative drops: `text` (the style and lyrics; the reference package's negative) or `sheet` (amplifies what the sheet adds). |
| `sample_sidecars` | `false` | Every AR sample is also written as a corpus track under `samples/sidecars/<sample id>/` (lossless audio, the exact codec tokens and NAR latents, the sheet and the request). Copied into `<root>/tracks/`, they are `replay_dirs` material. |
| `sheet_transcriber` | `sheetsage2` | Who writes a track's sheet when it has no `.abc`: `sheetsage2` or `refined` (see Sheets). |
| `refined_path` | `<MODELS_PATH>/ai_toolkit/SheetSage2-Refined` | SheetSage2-Refined's model folder. |
| `sheetsage_code`, `mert_path` | see above | The local folders. |

## Training in stages

One joint LoRA (`train_experts: both`) works, but the best results so far come from training the experts
separately: the NAR (the decoder) first, then the AR (the planner) in two steps, the sheet before the
music. Each stage is its own job with its own job name. Keep `semantic_head_path`, `level_fit`, `cot`,
`sheet_transcriber` and the dataset the same in every stage, so all three share one latent cache and the sheet
layout never changes.

### How the NAR LoRA carries into the AR stages

You want the AR stages' samples to render through your trained NAR, so you hear what the finished LoRA
will sound like. That is `other_expert_lora_path`, not `merge_nar_lora`: `merge_nar_lora` only folds the
Mothersuperior adapter's own file format into the base NAR and cannot read an ai-toolkit LoRA. Leave it
`false` in every stage. `other_expert_lora_path` holds the NAR LoRA frozen for the whole AR run, and every
checkpoint of that run carries it, so each checkpoint is a complete joint LoRA.

### Stage 1: NAR

```yaml
model_kwargs:
  semantic_head_path: "WannabeTokenizerV3.safetensors"
  cot: "full"
  sheet_transcriber: "refined" # or "sheetsage2"; the same in every stage
  train_experts: "nar"         # the AR loss is switched off
  merge_nar_lora: false
  abc_dropout: 0               # every item keeps its sheet; 0.5 (the default) also trains the off-mode prompt
```

Samples are the base AR's songs rendered by your NAR. `[Reconstruct]` samples are the cleanest check of this
stage: they render a real track's codec ids, so only the decoder is judged.

### Stage 2: AR, the sheet

```yaml
model_kwargs:
  semantic_head_path: "WannabeTokenizerV3.safetensors"
  cot: "full"
  sheet_transcriber: "refined"
  train_experts: "ar"
  ar_loss_on: "sheet"          # only the ABC sheet counts; abc_dropout is ignored, every item keeps its sheet
  ar_kl_weight: 1              # holds the music-token writing where it is while the sheet trains
  other_expert_lora_path: "/path/to/output/my_nar/my_nar_000002000.safetensors"   # stage 1's pick
  merge_nar_lora: false
  sample_ar_repetition_penalty: 1.2
```

Every track needs a sheet: SheetSage2's transcription or a manual `.abc`. With `ar_kl_weight: 0` the
music after the sheet is skipped entirely (the same loss for a fraction of the compute), but nothing then
holds the token writing in place.

### Stage 3: AR, the music

```yaml
model_kwargs:
  semantic_head_path: "WannabeTokenizerV3.safetensors"
  cot: "full"
  sheet_transcriber: "refined"
  train_experts: "ar"
  ar_loss_on: "tokens"         # only the codec tokens after the sheet count
  ar_kl_weight: 1              # anchored at base + the stage-2 LoRA, so the sheet writing is held
  base_lora_path: "/path/to/output/my_sheet/my_sheet_000001000.safetensors"       # stage 2's pick
  abc_dropout: 0
  merge_nar_lora: false
  sample_ar_repetition_penalty: 1.2
```

The stage-2 checkpoint is a joint file, so `base_lora_path` freezes its AR half under the new LoRA and holds
its NAR half too: no `other_expert_lora_path` is needed here. Don't also set it as
`network.pretrained_lora_path`. This run's checkpoints hold only the new music delta (plus the held NAR).
**The finished LoRA is the copy in `merged_exports/`** beside them: the sheet LoRA, the music LoRA and the NAR
in one file.

The NAR attends to the AR's keys and values over the prompt, and stage 1 trained it against the base AR. The
AR stages move those, and the KL leash limits how far. If renders get worse after stage 3, a short NAR pass
on top is possible (not yet tried): `train_experts: nar`, `base_lora_path` set to the stage-3 merged export,
whose AR half is then held automatically.

## Sample prompts

- **Semantic CFG**: a sample's `guidance_scale` is the CFG scale on the codec tokens, and **1 means off**.
  The reference package runs 1 (1.01 in `cot: off`) and suggests about 1.2 to experiment. A sample config
  without `guidance_scale` gets ai-toolkit's image default of 7, so set it explicitly. The ABC phase and
  the NAR never use CFG.
- **`[ABC]`**: a line `[ABC]` followed by a path to a `.abc` file sings that sheet instead of one the AR
  writes, a cover check during training.
- **`[Reconstruct]`**: a line `[Reconstruct]` followed by a path renders that track's codec ids with the
  NAR instead of ones the AR writes, a check of the decoder alone. An audio file is tokenized by the run's
  tokenizer exactly as the dataset reads it; a `.npy` holds codec ids (a minted track's `semantic.npy`).

```
pop, female vocal, piano, 90 bpm
[Lyrics]
[Verse 1]
...
[ABC]
/path/to/song.abc
```

## Diagnostics

[diagnostics.py](diagnostics.py) (`YuE2Lab`) measures the training signal without running the trainer:
base-model next-token CE on a token stream, AR -> NAR -> tokenizer round trips, NAR renders of real-song
tokens and token-stream statistics. It builds AR inputs through the trainer's own `_ar_inputs`, so the
numbers match training.
