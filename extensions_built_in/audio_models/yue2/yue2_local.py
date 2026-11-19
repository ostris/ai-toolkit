"""Training additions to the YuE2 trainer. They live in this subclass (registered in the base model's place by
yue2/__init__.py, same "yue2" arch) so yue2_model.py stays the base port; every override builds on super().

- Manual ABC sheets: ``song.abc`` beside ``song.wav`` is that track's training sheet instead of SheetSage2's
  transcription. Not in do_separation runs, whose stems need sheets of their own.
- ``model_kwargs.train_experts``: "both" (default, one joint LoRA), "nar" (decoder only, AR loss off) or "ar"
  (planner only: the NAR is skipped and the flow term is 0, so it needs ``ar_loss_weight`` > 0). A single-expert
  LoRA holds only that expert's keys, so two runs stack as two LoRAs and each half stops at its own best step.
- ``model_kwargs.other_expert_lora_path``: holds the expert a single-expert run does not train at a finished
  LoRA, frozen for the whole run (a joint file gives only that half), so a NAR run trains on the tuned AR's
  conditioning and samples play both halves. The run's checkpoints carry that half too, byte for byte, so each
  is a complete joint LoRA.
- Dataset pairing: YuE2 datasets fail fast on captionless tracks and stray sidecars (toolkit/dataset_pairing.py).
- Sheet samples: a sample prompt with an ``[ABC]`` line followed by a ``.abc`` path sings that sheet instead of
  one the AR writes, a cover check mid-training.
- Reconstruction samples: a sample prompt with a ``[Reconstruct]`` line followed by a path renders that track's codec
  ids with the NAR instead of ones the AR writes, a check of the decoder alone. An audio file is tokenized by the run's
  semantic head, read as the dataset reads it; a ``.npy`` holds codec ids (a minted track's or a sample sidecar's
  ``semantic.npy``). The NAR attends to the AR's keys and values over the whole prompt, so the render still depends on
  it: the sample's own style, lyrics and ``[ABC]`` sheet when it has them, else the track's (a song's caption ``.txt``
  and its sheet as the cache builds it, a minted track's request and sheet files). A track with no sheet, and any
  track in a cot "off" run, renders under the off-mode prompt. Cut to the sample duration; no CFG (nothing is
  sampled) and no corpus track (its ids are not ones the AR wrote).
- Sample CFG: the sample's ``guidance_scale`` is the semantic CFG scale, 1 = off (the reference package runs 1, 1.01
  in off mode, and suggests 1.2 to experiment). A second AR stream decodes the reference negative, the bare
  instruction plus the exact sheet, so CFG amplifies only what the style and lyrics add; each codec token samples
  from ``negative + scale * (prompt - negative)``. The ABC phase and the NAR never use it.
  ``model_kwargs.sample_cfg_negative: sheet`` negates the sheet instead (the same tags and lyrics as an off-mode
  prompt), amplifying what the sheet adds: melody, rhythm, its tempo.
- ``model_kwargs.sample_sidecars``: every AR sample is also written as a finished Yue2SemanticTokenizer corpus track,
  ``samples/sidecars/<sample id>/`` (lossless audio, the exact codec tokens and NAR latents rendered, the sheet and
  the request), ready to copy into ``datasets/minted_ours/tracks/``.
- ``model_kwargs.ar_loss_on``: "all" (default), "sheet" or "tokens", the AR targets the next-token loss counts.
  "sheet" trains only the ABC sheet (its text and ABC_END; abc_dropout is ignored, so every item keeps its sheet),
  "tokens" only the music after it. The KL leash still
  covers every position, so it holds whatever the loss leaves out.
- ``model_kwargs.base_lora_path``: a finished LoRA held frozen on the expert(s) this run trains, with the new LoRA
  stacked on top, so a staged run's next stage builds on the last (KL then anchors at base + that LoRA). Its
  checkpoints hold only the new delta and are stamped with the base they need; ``merged_exports/`` beside them
  gets each step as one complete LoRA (rank-concatenated, exact).
- ``model_kwargs.replay_dirs`` (+ ``replay_ratio``, default 0.2 when dirs are given, ``replay_weight`` 1.0): exact-token
  replay. Corpora in the Yue2SemanticTokenizer minted layout (``<root>/tracks/<pid>/semantic.npy`` = codes the AR wrote
  itself, ``request.json`` with style / lyrics / cot, ``score.abc``). Real songs carry a tokenizer's approximate labels;
  with probability ``replay_ratio`` a step adds one replay track's next-token CE (its own prompt and sheet, the sheet
  dropped as abc_dropout drops real sheets), weighted ``replay_weight`` against the real term, so the AR keeps its own
  codes while it learns the new songs. Logged as ``loss/ar_replay_ce``. AR term only; the NAR is untouched.
- ``model_kwargs.semantic_head_path`` may point at a Yue2SemanticTokenizer end-to-end pair
  (``runs/<run>/best/head.safetensors``, its encoder beside it): ``src/tokenizer.py`` dispatches on the file's
  ``head.kind`` metadata, and the latent cache is keyed by ``<run>-<best|last>`` (every pair's file is head.safetensors).
"""
import glob
import hashlib
import inspect
import json
import math
import os
import random
import re
from typing import List, Optional, Tuple

import numpy as np

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from toolkit.basic import UnusableFileError
from toolkit.metadata import add_model_hash_to_meta
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds

from .src.model import ABC_END, CODEC_SIZE, FRAMES_PER_SECOND
from .src.pipeline import YuE2Pipeline
from .src.vae import SAMPLE_RATE
from .yue2_model import AI_TOOLKIT_DIR, YuE2AudioModel as UpstreamYuE2AudioModel, apply_level_fit, parse_caption, resolve_ai_toolkit_file

# upstream methods the overrides below extend: a rename upstream has to stop here, not silently skip a feature
_EXTENDED = (
    "load_model", "encode_audio", "_transcribe_sheets", "pop_encode_warnings", "get_loss_target", "get_noise_prediction",
    "get_prompt_embeds", "generate_single_audio", "_ar_losses", "get_additional_loss", "_ar_inputs", "get_latent_space_version",
    "_get_semantic_tokenizer",
)
_missing = [name for name in _EXTENDED if name not in UpstreamYuE2AudioModel.__dict__]
if _missing:
    raise ImportError(f"upstream YuE2AudioModel no longer defines {_missing}: update yue2_local.py to match")
# sample CFG hands a negative prefix to the pipeline's music phase
if "negative_embeds" not in inspect.signature(YuE2Pipeline.generate_codec_tokens).parameters:
    raise ImportError("src/pipeline.py's generate_codec_tokens no longer takes negative_embeds: update yue2_local.py to match")

# model_kwargs.train_experts -> the expert classes the LoRA wraps
TRAIN_EXPERTS = {"both": ["YuE2AR", "YuE2NAR"], "ar": ["YuE2AR"], "nar": ["YuE2NAR"]}
# saved LoRA key prefix per expert (upstream's convert_lora_weights_before_save layout)
SAVED_LORA_PREFIX = {"nar": "diffusion_model.", "ar": "text_encoders."}
AR_LOSS_ON = ("all", "sheet", "tokens")
# model_kwargs.sample_cfg_negative: what the CFG negative prompt drops
CFG_NEGATIVES = ("text", "sheet")
# sample prompt sections whose next non-empty line is a file path (see split_paths)
PATH_SECTIONS = ("abc", "reconstruct")
# checkpoint metadata of a run over a base_lora_path: the sha256 of the base file its delta needs, and its name
BASE_MARKER, BASE_NAME = "yue2_base_lora_sha256", "yue2_base_lora"

# a chord symbol in the ABC body is a quoted string starting with a note letter ("Am7", "F#/C#");
# annotations ("^text") do not match, and header fields / SheetSage2's "% Section" lines are skipped
_ABC_CHORD = re.compile(r'"[A-G][^"]*"')
_ABC_NON_BODY = re.compile(r"^\s*(%|[A-Za-z]:)")


def abc_sidecar_path(audio_path: str) -> str:
    """``song.abc`` beside ``song.wav``, the way ``song.txt`` carries its caption."""
    return os.path.splitext(audio_path)[0] + ".abc"


def read_abc(path: str) -> Optional[str]:
    """A hand-written sheet as SheetSage2 would serialize it (one trailing newline), or None when it is blank."""
    with open(path, "r", encoding="utf-8") as f:
        abc = f.read().strip()
    return abc + "\n" if abc else None


def read_abc_sidecar(audio_path: str) -> Optional[str]:
    """The manual sheet's text, or None when there is none (or it is blank) and SheetSage2 transcribes."""
    path = abc_sidecar_path(audio_path)
    return read_abc(path) if os.path.isfile(path) else None


def lora_metadata(path: str) -> dict:
    if not path.endswith(".safetensors"):
        return {}
    with safe_open(path, "pt") as f:
        return f.metadata() or {}


def abc_has_chords(abc: str) -> bool:
    return any(_ABC_CHORD.search(line) for line in abc.splitlines() if not _ABC_NON_BODY.match(line))


def _section_name(line: str) -> Optional[str]:
    """``[Name]`` -> "name" for a bracketed line, else None."""
    s = line.strip()
    return s[1:-1].strip().lower() if len(s) > 2 and s[0] == "[" and s[-1] == "]" else None


def split_paths(prompt: str) -> Tuple[str, dict]:
    """A sample prompt can name files by section: an ``[ABC]`` line (a sheet to sing) or a ``[Reconstruct]`` line (a
    track for the NAR to render), then the file's path on the next non-empty line. Returns the prompt without those
    lines, and {section: path} per section present (None when the next non-empty line is another section)."""
    lines = prompt.splitlines()
    drop, paths = set(), {}
    for i, line in enumerate(lines):
        name = _section_name(line)
        if name not in PATH_SECTIONS or name in paths:
            continue
        drop.add(i)
        j = next((j for j in range(i + 1, len(lines)) if lines[j].strip()), None)
        paths[name] = None
        if j is not None and _section_name(lines[j]) is None:
            drop.add(j)
            paths[name] = lines[j].strip().strip("\"'")
    return ("\n".join(l for i, l in enumerate(lines) if i not in drop) if drop else prompt), paths


def load_codes(path: str) -> np.ndarray:
    """A ``.npy`` of codec ids -> [T] int64. Anything else (latents, a batch, ids past the codebook) is refused here,
    where an out-of-range id would otherwise fault the embedding lookup on the GPU."""
    codes = np.load(path)
    if codes.ndim != 1 or codes.dtype.kind not in "iu" or (codes.size and not 0 <= codes.min() <= codes.max() < CODEC_SIZE):
        raise ValueError(f"{path} is not a 1-D array of codec ids in [0, {CODEC_SIZE})")
    return codes.astype(np.int64)


def _recording(fn, captured: dict, name: str):
    """Wrap a pipeline method so its arguments and result are kept (for a sample's sidecars)."""

    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        captured[name] = (args, kwargs, out)
        return out

    return wrapped


class YuE2AudioModel(UpstreamYuE2AudioModel):
    # the caption is the tags + lyrics the AR is prompted with: a track without one, or a sidecar that pairs with
    # no track, is a naming slip that must stop the run (a dataset's strict_pairing: false opts out)
    strict_dataset_pairing = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.model_config.model_kwargs
        self.train_experts = str(kw.get("train_experts", "both"))
        if self.train_experts not in TRAIN_EXPERTS:
            raise ValueError(f"model_kwargs.train_experts must be one of {list(TRAIN_EXPERTS)}")
        self.target_lora_modules = TRAIN_EXPERTS[self.train_experts]
        if self.train_experts == "nar":
            self.ar_loss_weight = 0.0  # no AR LoRA for the next-token loss to train
        elif self.train_experts == "ar" and self.ar_loss_weight <= 0:
            raise ValueError("model_kwargs.train_experts 'ar' trains through the AR loss only; set ar_loss_weight > 0")
        if self.train_experts != "both":
            self.ar_lr_multiplier = 1.0  # relative to the NAR's rate, so moot when one expert trains
        self.other_expert_lora_path = kw.get("other_expert_lora_path", None)
        if self.other_expert_lora_path and self.train_experts == "both":
            raise ValueError("model_kwargs.other_expert_lora_path needs train_experts 'ar' or 'nar'")
        self.other_expert_lora = None
        self._held_weights = {}  # the held half as saved in its file (saved key names), written into checkpoints
        self.base_lora_path = kw.get("base_lora_path", None)
        self.base_lora = None
        self._base_weights = {}  # the base as saved in its file; never in checkpoints, only in merged exports
        self._base_id = self._base_name = None
        self._sheet_paths: List[str] = []  # the latent cacher's source paths for the batch being encoded
        self._local_warnings: List[str] = []
        self.sample_sidecars = bool(kw.get("sample_sidecars", False))
        self._pending_track = None
        self._reconstruct_sources = {}  # (path, mtime, size) of a [Reconstruct] file -> what _reconstruct_source read
        if self.sample_sidecars:
            # a sample's file name is settled only when it is saved, so its track is written from the after-save hook
            self.add_after_sample_image_hook(self._write_sample_track)
        # sample CFG (the per-sample guidance_scale): the negative prompt drops the style and lyrics ("text", the
        # reference package's) or the sheet ("sheet": the same tags and lyrics as an off-mode prompt)
        self.sample_cfg_negative = str(kw.get("sample_cfg_negative", "text"))
        if self.sample_cfg_negative not in CFG_NEGATIVES:
            raise ValueError(f"model_kwargs.sample_cfg_negative must be one of {list(CFG_NEGATIVES)}")
        if self.sample_cfg_negative == "sheet" and self.cot == "off":
            raise ValueError("model_kwargs.sample_cfg_negative 'sheet' drops the sheet, which cot 'off' never writes")
        self._cfg_noted = set()  # (scale, negative, cot) already announced
        self.ar_loss_on = str(kw.get("ar_loss_on", "all"))
        if self.ar_loss_on not in AR_LOSS_ON:
            raise ValueError(f"model_kwargs.ar_loss_on must be one of {list(AR_LOSS_ON)}")
        if self.ar_loss_on != "all" and self.ar_loss_weight <= 0:
            raise ValueError("model_kwargs.ar_loss_on picks the AR loss's targets, so it needs ar_loss_weight > 0")
        if self.ar_loss_on == "sheet":
            if self.cot == "off":
                raise ValueError("model_kwargs.ar_loss_on 'sheet' trains the ABC sheet, which cot 'off' never writes")
            if self.abc_dropout > 0:
                # a dropped sheet leaves no sheet to train (upstream reads abc_dropout per item, so this holds all run;
                # in a joint run the NAR then trains on sheet conditioning only too)
                print_acc(f"YuE2: ar_loss_on 'sheet' ignores abc_dropout {self.abc_dropout}: every item keeps its sheet")
            self.abc_dropout = 0.0
        self._long_sheet_noted = False
        # exact-token replay (see the module docstring): minted-layout corpora whose codes the AR wrote itself
        dirs = kw.get("replay_dirs", None) or []
        self.replay_dirs = [dirs] if isinstance(dirs, str) else [str(d) for d in dirs]
        self.replay_ratio = float(kw.get("replay_ratio", 0.2 if self.replay_dirs else 0.0))
        self.replay_weight = float(kw.get("replay_weight", 1.0))
        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError("model_kwargs.replay_ratio must be within [0, 1]")
        if self.replay_ratio > 0 and not self.replay_dirs:
            raise ValueError("model_kwargs.replay_ratio > 0 needs replay_dirs (corpora in the minted layout)")
        if self.replay_ratio > 0 and self.ar_loss_weight <= 0:
            raise ValueError("exact-token replay trains through the AR loss: set ar_loss_weight > 0 (train_experts 'ar' or 'both')")
        for root in self.replay_dirs:
            if not os.path.isdir(os.path.join(root, "tracks")):
                raise ValueError(f"model_kwargs.replay_dirs: {root} has no tracks/ folder (minted layout expected)")
        self._replay_pool: Optional[List[str]] = None
        self._pending_replay_ce = None

    # ------------------------------------------------------------------ loading
    def load_model(self):
        super().load_model()
        other = {"ar": "nar", "nar": "ar"}.get(self.train_experts)
        if self.base_lora_path:
            path = resolve_ai_toolkit_file(self.base_lora_path, component="yue2 base lora")
            self.base_lora, self._base_weights = self._load_frozen_lora(path, self.train_experts)
            with open(path, "rb") as f:
                self._base_id = hashlib.sha256(f.read()).hexdigest()
            self._base_name = os.path.basename(path)
            with safe_open(path, "pt") as f:
                joint = other and any(k.startswith(SAVED_LORA_PREFIX[other]) for k in f.keys())
            if joint and not self.other_expert_lora_path:
                # a joint base (a previous stage's complete LoRA) also holds the other half, so it stays in the song
                self.other_expert_lora_path = path
                print_acc(f"YuE2: other_expert_lora_path unset, so the base's {other.upper()} half is held too")
        if self.other_expert_lora_path:
            path = resolve_ai_toolkit_file(self.other_expert_lora_path, component="yue2 other expert lora")
            # _held_weights is set only after the load: convert_lora_weights_before_load drops the keys it holds
            self.other_expert_lora, self._held_weights = self._load_frozen_lora(path, other)

    def _load_frozen_lora(self, path: str, experts: str):
        """Frozen LoRA on ``experts`` (a train_experts value) from a saved YuE2 LoRA, single-expert or joint, built
        like the trainer's own network so it loads the way a resume would; a LoRA the trainer applies later stacks on
        top of it. Returns (network, the file's weights for those experts under their saved key names)."""
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        if BASE_MARKER in lora_metadata(path):
            raise ValueError(
                f"{path} was trained over a base_lora_path and holds only its own delta: use the copy in "
                f"merged_exports/ beside it, which includes its base"
            )
        prefixes = tuple(p for e, p in SAVED_LORA_PREFIX.items() if experts in (e, "both"))
        weights = {k: v for k, v in load_file(path).items() if k.startswith(prefixes)}
        if not weights:
            raise ValueError(f"{path} has no {experts.upper()} weights ({'/'.join(prefixes)}*) to hold fixed")
        odd = next((k for k in weights if not k.endswith(("lora_A.weight", "lora_B.weight"))), None)
        if odd:
            raise ValueError(f"{path}: only plain LoRA pairs can be held fixed, found {odd}")
        # peft-format YuE2 LoRAs always train at alpha = rank (scale 1): a lower-rank module zero-pads to the max exactly
        rank = max(int(v.shape[0]) for k, v in weights.items() if k.endswith("lora_A.weight"))
        model = self.get_model_to_train()
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model,
            lora_dim=rank,
            multiplier=1.0,
            alpha=rank,
            train_unet=True,
            train_text_encoder=False,
            network_config=NetworkConfig(type="lora", linear=rank, linear_alpha=rank),
            network_type="lora",
            transformer_only=True,
            is_transformer=self.is_transformer,
            base_model=self,
            target_lin_modules=TRAIN_EXPERTS[experts],
            is_assistant_adapter=True,
        )
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        unmatched = network.load_weights(weights)
        if unmatched:
            # load_weights only prints strays; a fixed LoRA that silently loaded nothing would train against the base
            raise ValueError(f"{path}: {len(unmatched)} {experts.upper()} LoRA keys matched no module, e.g. {next(iter(unmatched))}")
        # frozen: it shapes what the trained LoRA sees but is never trained
        for param in network.parameters():
            param.requires_grad_(False)
        network.eval()
        network.is_active = True
        self.print_and_status_update(f"YuE2: {experts.upper()} held at {os.path.basename(path)} (rank {rank}, frozen)")
        return network, weights

    def convert_lora_weights_before_save(self, state_dict):
        # a single-expert run's checkpoints also carry the held half, so each is a complete joint LoRA
        return {**super().convert_lora_weights_before_save(state_dict), **self._held_weights}

    def convert_lora_weights_before_load(self, state_dict):
        # resuming those checkpoints: the held half belongs to the frozen network, not the one being trained
        return super().convert_lora_weights_before_load({k: v for k, v in state_dict.items() if k not in self._held_weights})

    def save_lora(self, save_dict, file, metadata):
        """Every checkpoint save (network_mixins' hook). Over a base_lora_path the checkpoint keeps only this run's
        delta, so a resume loads exactly what trains, stamped with the base it needs; merged_exports/ gets the same
        step as one complete LoRA."""
        if self._base_weights:
            metadata = {**metadata, BASE_MARKER: self._base_id, BASE_NAME: self._base_name}
        save_file(save_dict, file, metadata)
        if self._base_weights:
            self._write_merged_export(save_dict, file, metadata)

    def _write_merged_export(self, save_dict, file, metadata):
        """base + delta as one LoRA: a module both hold is rank-concatenated, exact at scale 1 (peft, alpha = rank):
        [B_base | B_new] @ [A_base ; A_new] = B_base A_base + B_new A_new. Never a resume candidate, being in a
        subfolder, and kept in step with the checkpoints the trainer keeps."""
        if any(not k.endswith(("lora_A.weight", "lora_B.weight")) for k in save_dict):
            print_acc("YuE2: no merged export (only a plain LoRA network concatenates): load the base and the checkpoint as two LoRAs")
            return
        merged = dict(save_dict)
        for k, base in self._base_weights.items():
            new = merged.get(k)
            merged[k] = base if new is None else torch.cat([base.to(new.dtype), new], dim=0 if k.endswith("lora_A.weight") else 1)
        root, name = os.path.split(file)
        folder = os.path.join(root, "merged_exports")
        os.makedirs(folder, exist_ok=True)
        meta = {k: v for k, v in metadata.items() if k not in (BASE_MARKER, BASE_NAME)}
        meta["yue2_merged_base"] = self._base_name
        save_file(merged, os.path.join(folder, name), add_model_hash_to_meta(merged, meta))
        for old in os.listdir(folder):
            if old.endswith(".safetensors") and not os.path.exists(os.path.join(root, old)):
                os.remove(os.path.join(folder, old))  # its checkpoint was rotated out

    def load_lora(self, file):
        """The trainer's resume and pretrained_lora_path loads (network_mixins' hook, str paths only; the frozen
        LoRAs load from dicts). A checkpoint trained over a base is only a delta, so it loads over that same base,
        and a run over a base loads only its own checkpoints."""
        meta = lora_metadata(file)
        if meta.get(BASE_MARKER) != self._base_id:
            raise ValueError(
                f"{os.path.basename(file)} was trained over {meta.get(BASE_NAME) or 'no base'}, but this run's "
                f"base_lora_path is {self._base_name or 'unset'}. A stage over a base needs a new job name (a resume "
                "would otherwise pick up the previous stage's checkpoint) and must not repeat the base as "
                "network.pretrained_lora_path; a delta checkpoint needs its own base, or use its merged_exports/ copy."
            )
        return load_file(file) if file.endswith(".safetensors") else torch.load(file, map_location="cpu")

    # ------------------------------------------------------------------ manual ABC sheets
    def set_encode_source_paths(self, paths: List[str]):
        """The latent cacher's paths for the next encode_audio batch, so its sheets can come from manual sidecars."""
        self._sheet_paths = list(paths)

    def encode_audio(self, audio_tensor, device=None, dtype=None):
        try:
            return super().encode_audio(audio_tensor, device=device, dtype=dtype)
        finally:
            self._sheet_paths = []  # never outlives the batch it was set for

    def _transcribe_sheets(self, wavs):
        """Items whose track has a manual ``.abc`` sidecar use it verbatim; the rest go through upstream's
        SheetSage2 path, which never loads when every item has a sidecar."""
        paths = self._sheet_paths
        manual = [read_abc_sidecar(paths[i]) if i < len(paths) else None for i in range(len(wavs))]
        if all(abc is None for abc in manual):
            return super()._transcribe_sheets(wavs)
        if self.do_separation:
            raise ValueError(
                "manual ABC sidecars are not supported with do_separation (each stem needs its own sheet): "
                "remove the .abc files or turn separation off"
            )
        auto = [i for i, abc in enumerate(manual) if abc is None]
        auto_sheets = super()._transcribe_sheets([wavs[i] for i in auto]) if auto else None
        rows = []
        for i, abc in enumerate(manual):
            if abc is None:
                row = auto_sheets[auto.index(i)]
                rows.append(row[row >= 0])
            else:
                self._check_sidecar_layout(abc, paths[i])
                self._local_warnings.append("sheet taken from the manual ABC sidecar")
                rows.append(torch.tensor(self.tokenizer.encode_abc(abc), dtype=torch.int32))
        sheets = torch.full((len(rows), max(len(r) for r in rows)), -1, dtype=torch.int32)
        for i, row in enumerate(rows):
            sheets[i, : len(row)] = row
        return sheets

    def _check_sidecar_layout(self, abc: str, audio_path: str):
        """A manual sheet has to match the run's cot flavour, as upstream's abc_mode guards cached sheets: chords in
        a melody run would train the planner on a layout it never writes. A chordless sheet in a full run is only
        noted, since a song can legitimately have no chords."""
        has_chords = abc_has_chords(abc)
        if self.cot == "melody" and has_chords:
            raise ValueError(
                f"The ABC sidecar for {audio_path} has chord symbols but model_kwargs.cot is 'melody': "
                "remove the quoted chords or train with cot: full"
            )
        if self.cot == "full" and not has_chords:
            self._local_warnings.append("ABC sidecar has no chord symbols although cot is 'full' (fine for a chordless song)")

    def pop_encode_warnings(self) -> List[str]:
        warnings, self._local_warnings = self._local_warnings, []
        return warnings + super().pop_encode_warnings()

    def get_file_cache_key(self, path: str) -> Optional[str]:
        """Per-file latent-cache material: the manual sheet is baked into the cached abc_ids, so adding or editing
        one rebuilds that file's latent. None without a sidecar keeps existing caches valid."""
        abc = read_abc_sidecar(path) if self.cot != "off" else None
        return hashlib.md5(abc.encode("utf-8")).hexdigest() if abc else None

    def get_dataset_sidecar_paths(self, path: str) -> List[str]:
        """Sidecars a track may have besides its caption, for the dataset pairing check."""
        return [abc_sidecar_path(path)]

    # ------------------------------------------------------------------ sheet samples
    def get_prompt_embeds(self, prompt):
        # a sample's [ABC] and [Reconstruct] lines are not part of the prompt the AR reads; generate_single_audio feeds
        # the sheet and the track
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        return super().get_prompt_embeds([split_paths(p)[0] for p in prompts])

    def generate_single_audio(self, pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra):
        # per-sample pipeline overrides, all removed again afterwards: a [Reconstruct] track's codec ids and sheet stand
        # in for the ones the AR would write, so does a supplied [ABC] sheet, CFG decodes a negative prompt beside the
        # codec tokens, and with sample_sidecars the sheet and the render are recorded on the way through
        overrides = {}
        sheet = None
        recon = self._sample_reconstruction(gen_config.prompt)
        if recon is not None:
            conditional_embeds, abc_ids, codes = recon  # the prompt the track renders under
            overrides["generate_abc_tokens"] = lambda *a, **kw: list(abc_ids)
            overrides["generate_codec_tokens"] = lambda prefix, max_tokens, **kw: codes[:max_tokens]  # cut to the duration
        else:
            sheet = self._sample_sheet(gen_config.prompt)
            if sheet is not None:
                overrides["generate_abc_tokens"] = lambda *a, **kw: list(sheet[0])
        captured = {}
        # a reconstruction samples no codec tokens, so CFG has nothing to guide; nor is it a corpus track, since its
        # ids are not ones the AR wrote
        scale = 1.0 if recon is not None else self._sample_cfg_scale(gen_config)
        sidecars = self.sample_sidecars and recon is None
        record = ["synthesize"] if sidecars else []
        if sidecars or scale != 1.0:
            record.append("generate_abc_tokens")  # the CFG negative sings the positive branch's exact sheet ids
        for name in record:
            overrides[name] = _recording(overrides.get(name, getattr(pipeline, name)), captured, name)
        if scale != 1.0:
            music = pipeline.generate_codec_tokens

            def guided(prefix, **kw):
                abc_ids = captured["generate_abc_tokens"][2] if "generate_abc_tokens" in captured else []  # off: none
                negative = self._cfg_negative_embeds(gen_config.prompt, abc_ids).to(prefix.dtype)
                return music(prefix, negative_embeds=negative, cfg_scale=scale, **kw)

            overrides["generate_codec_tokens"] = guided
        for name, fn in overrides.items():
            setattr(pipeline, name, fn)
        try:
            audio = super().generate_single_audio(pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra)
        finally:
            for name in overrides:
                delattr(pipeline, name)  # back to the pipeline's own methods
        if sidecars:
            self._pending_track = (gen_config, captured, audio, None if sheet is None else sheet[1], scale)
        return audio

    def _sample_cfg_scale(self, gen_config) -> float:
        """The sample's guidance_scale as the semantic CFG scale, 1 = off. The reference package runs 1 (1.01 in off
        mode) and suggests 1.2 to experiment; a sample config without guidance_scale gets ai-toolkit's image default
        of 7. A value outside the reference's 0-20 is said loudly and the sample runs without CFG (a sample never
        ends a run)."""
        scale = gen_config.guidance_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 <= scale <= 20:
            print_acc(f"YuE2 sample: guidance_scale {scale!r} is not a CFG scale in 0-20, sampling without CFG")
            return 1.0
        scale = float(scale)
        note = (scale, self.sample_cfg_negative, self.cot)
        if scale != 1.0 and note not in self._cfg_noted:
            self._cfg_noted.add(note)
            dropped = "the sheet" if self.sample_cfg_negative == "sheet" else "the style and lyrics"
            print_acc(f"YuE2 samples: CFG {scale:g} on the codec tokens (cot {self.cot}; the negative drops {dropped}; "
                      f"the reference default is {1.01 if self.cot == 'off' else 1:g}, 1.2 to experiment)")
        return scale

    def _cfg_negative_embeds(self, prompt: str, abc_ids: List[int]) -> torch.Tensor:
        """The CFG negative as AR embeddings [1, L, H]: the reference package's (the bare instruction plus the exact
        sheet, so CFG amplifies the style and lyrics), or with sample_cfg_negative 'sheet' the same tags and lyrics as
        an off-mode prompt (so CFG amplifies what the sheet adds)."""
        if self.sample_cfg_negative == "sheet":
            parsed = parse_caption(split_paths(prompt)[0])
            ids = self.tokenizer.prefix_ids(parsed["style"], parsed["lyrics"], cot="off")
        else:
            ids = self.tokenizer.negative_prefix_ids(self.cot, abc_ids)
        return self.model.ar.embed(torch.tensor([ids], device=self.device_torch))

    def _sample_sheet(self, prompt: str):
        """(sheet ids, sheet text) named by a sample prompt's [ABC] lines, or None. A sample must never end a run
        over a typo: problems are said loudly and the AR writes its own sheet."""
        sheet_path = split_paths(prompt)[1].get("abc")
        if sheet_path is None:
            return None
        if self.cot == "off":
            print_acc("YuE2 sample: cot is 'off', so the [ABC] sheet is ignored")
            return None
        abc = read_abc(sheet_path) if os.path.isfile(sheet_path) else None
        if abc is None:
            print_acc(f"YuE2 sample: sheet {sheet_path} not found or blank, the AR writes its own")
            return None
        if self.cot == "melody" and abc_has_chords(abc):
            print_acc(f"YuE2 sample: {os.path.basename(sheet_path)} has chords but cot is 'melody'")
        print_acc(f"YuE2 sample: singing the sheet {os.path.basename(sheet_path)}")
        return self.tokenizer.encode_abc(abc), abc

    def _write_sample_track(self, img_num, total_imgs):
        """After-save hook: the sample just saved, as a Yue2SemanticTokenizer corpus track in
        samples/sidecars/<job>_<sample file>__<cot>_s<seed>/, in the corpus's write_track layout: audio.flac (lossless,
        whatever the sample format), semantic.npy (the exact codec ids the NAR rendered), latent.npy (its latents
        [T, 64]), abc_tokens.npy + score.abc (the sheet the AR sang from) and request / item / result.json. The NAR
        rendered exactly these tokens into exactly this audio, so the pair is valid: copy the folder into
        datasets/minted_ours/tracks/ (any name), then run the corpus's features.py and index_corpus.py."""
        import json

        import numpy as np
        import soundfile as sf

        if self._pending_track is None:
            return
        (gen_config, captured, audio, sheet_text, cfg_scale), self._pending_track = self._pending_track, None
        if "synthesize" not in captured:
            print_acc("YuE2 sample: the sampler no longer calls pipeline.synthesize, so no corpus track (update yue2_local.py)")
            return
        (prefix, codec), render_kw, latents = captured["synthesize"]
        if len(codec) < FRAMES_PER_SECOND:
            print_acc("YuE2 sample: under a second of music, no corpus track written")
            return
        abc_ids = [int(v) for v in captured["generate_abc_tokens"][2]] if "generate_abc_tokens" in captured else []
        abc_text = sheet_text or (self.tokenizer.decode(abc_ids) if abc_ids else None)
        parsed = parse_caption(split_paths(gen_config.prompt)[0])
        style, lyrics = parsed["style"], parsed["lyrics"]
        job = os.path.basename(os.path.dirname(os.path.normpath(gen_config.output_folder)))
        sample = os.path.basename(gen_config.get_image_path(img_num))  # the saved sample's own file name
        pid = f"{job}_{os.path.splitext(sample)[0].replace('__', '_')}__{self.cot}_s{render_kw['seed']}"
        d = os.path.join(gen_config.output_folder, "sidecars", pid)
        os.makedirs(d, exist_ok=True)
        nar_adapter = os.path.basename(self.nar_lora_path) if self.merge_nar_lora else None
        sf.write(os.path.join(d, "audio.flac"), audio[0].float().clamp(-1, 1).T.numpy(), SAMPLE_RATE, subtype="PCM_24")
        np.save(os.path.join(d, "semantic.npy"), np.asarray(codec, dtype=np.int32))
        np.save(os.path.join(d, "latent.npy"), latents[0].float().cpu().numpy())
        np.save(os.path.join(d, "abc_tokens.npy"), np.asarray(abc_ids, dtype=np.int32))
        if abc_text:
            with open(os.path.join(d, "score.abc"), "w", encoding="utf-8") as f:
                f.write(abc_text)
        request = {
            "style": style, "lyrics": lyrics, "cot": self.cot, "seed": render_kw["seed"], "abc": abc_text, "id": pid,
            "cfg_scale": cfg_scale,  # the reference SongRequest field; CFG-guided codes are not the AR's plain sampling
        }
        item = {
            "source": sample, "caption": style, "family": "ours", "genre": style.split(",")[0].strip()[:40],
            "instrumental": lyrics.strip().lower() in ("", "[instrumental]"), "origin": "ai-toolkit sample",
            "abc_source": "sample [ABC] sheet" if sheet_text else ("ar" if abc_ids else None),
            "head": None, "nar_lora": nar_adapter, "lora": job, "base_lora": self._base_name, "train_experts": self.train_experts,
            "cfg_negative": self.sample_cfg_negative if cfg_scale != 1.0 else None,
        }
        result = {
            "status": "complete", "frames": len(codec), "audio_seconds": len(codec) / FRAMES_PER_SECOND,
            "nar_lora": nar_adapter, "steps": render_kw["steps"], "prefix_tokens": int(prefix.shape[1]),
        }
        # result.json last: the corpus reads its presence as "track complete"
        for name, obj in (("request.json", request), ("item.json", item), ("result.json", result)):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
        print_acc(f"YuE2 sample: corpus track -> {d}")

    # ------------------------------------------------------------------ reconstruction samples
    def _sample_reconstruction(self, prompt: str):
        """(prompt embeds, sheet ids, codec ids) of the track a sample prompt's [Reconstruct] lines name, or None for an
        ordinary sample. The sample's own style and lyrics (when it has any) and its [ABC] sheet stand in for the
        track's; a track with no sheet, and any track in a cot "off" run, renders under the off-mode prompt, as
        abc_dropout trains it. A sample must never end a run over a typo: an unusable file is said loudly and the AR
        samples as usual."""
        text, paths = split_paths(prompt)
        if "reconstruct" not in paths:
            return None
        path = paths["reconstruct"]
        if not path or not os.path.isfile(path):
            print_acc(f"YuE2 sample: reconstruction track {path or '(none named)'} not found, the AR samples as usual")
            return None
        source = self._reconstruct_source(path)
        if source is None:
            return None
        codes, style, lyrics, cot, abc_ids = source
        parsed = parse_caption(text)
        if parsed["style"] or parsed["lyrics"]:
            style, lyrics = parsed["style"], parsed["lyrics"]
        sheet = self._sample_sheet(prompt)
        if sheet is not None:
            cot, abc_ids = self.cot, sheet[0]
        if self.cot == "off" or not abc_ids:
            cot, abc_ids = "off", []
        ids = torch.tensor([self.tokenizer.prefix_head_ids(style, lyrics, cot=cot)], device=self.device_torch)
        label = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))  # minted files share a name
        empty = "" if style or lyrics else ", no style or lyrics"
        print_acc(f"YuE2 sample: reconstructing {label} ({len(codes) / FRAMES_PER_SECOND:.1f}s of codec ids, cot {cot}{empty})")
        return PromptEmbeds(self.model.ar.embed(ids), attention_mask=torch.ones_like(ids)), abc_ids, codes

    def _reconstruct_source(self, path: str):
        """A [Reconstruct] file -> (codec ids, style, lyrics, cot, sheet ids), memoized per file version so the semantic
        head and SheetSage2 run once per track, not every sampling round; None (said loudly) when the file cannot be
        used. A minted track's ``semantic.npy`` comes with the prompt it was minted from, any other ``.npy`` is bare
        codec ids with no prompt of its own, and anything else is a song (see _read_song)."""
        from av.error import FFmpegError

        stat = os.stat(path)
        key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._reconstruct_sources:
            name, folder = os.path.basename(path), os.path.dirname(path)
            try:
                if not name.lower().endswith(".npy"):
                    source = self._read_song(path)
                elif name == "semantic.npy" and os.path.isfile(os.path.join(folder, "request.json")):
                    codes, style, lyrics, cot, abc_ids = self._minted_track(folder)
                    source = (codes.tolist(), style, lyrics, cot, abc_ids)
                else:
                    source = (load_codes(path).tolist(), "", "", "off", [])
                if not source[0]:
                    raise ValueError("no codec ids")
            except (OSError, ValueError, FFmpegError) as e:
                print_acc(f"YuE2 sample: cannot reconstruct {name} ({type(e).__name__}: {e}), the AR samples as usual")
                return None
            self._reconstruct_sources[key] = source
        return self._reconstruct_sources[key]

    def _read_song(self, path: str):
        """A song read the way the latent cache reads a training one -> (codec ids, style, lyrics, cot, sheet ids):
        decoded and resampled as the dataset loads it, its loudness fitted as the cache fits it (level_fit), ids
        from the run's semantic head, the sheet (cot full/melody) from its ABC sidecar else SheetSage2, the style and
        lyrics from the caption ``.txt`` beside it."""
        import torchaudio

        from toolkit.audio.decode import load_audio
        from toolkit.dataloader_mixins import waveform_to_stereo

        name = os.path.basename(path)
        caption_path = os.path.splitext(path)[0] + ".txt"
        caption = ""
        if os.path.isfile(caption_path):
            with open(caption_path, encoding="utf-8") as f:
                caption = f.read()
        parsed = parse_caption(caption)
        wav, sr = load_audio(path)
        wav = waveform_to_stereo(wav)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = apply_level_fit(wav, self.level_fit, self.level_band())
        self._status_update(f"YuE2: reading {name} for reconstruction")
        tokenizer = self._get_semantic_tokenizer()
        if tokenizer.device != self.device_torch:
            tokenizer.to(self.device_torch)
        codes = tokenizer.tokenize(wav, SAMPLE_RATE).tolist()
        abc_ids: List[int] = []
        if self.cot != "off":
            self.set_encode_source_paths([path])  # an ABC sidecar beside it is its sheet, as the cache takes it
            try:
                row = self._transcribe_sheets([wav])[0]
                abc_ids = row[row >= 0].tolist()
            except (UnusableFileError, ValueError) as e:
                print_acc(f"YuE2 sample: no sheet for {name} ({e}), it renders under the off-mode prompt")
            finally:
                self._sheet_paths = []
            for warning in self.pop_encode_warnings():
                print_acc(f"YuE2 sample: {name}: {warning}")
        return codes, parsed["style"], parsed["lyrics"], self.cot, abc_ids

    # ------------------------------------------------------------------ staged AR training
    def _ar_losses(self, ar_embeds, ids, prefix, total, whole_song):
        """ar_loss_on "sheet" / "tokens": next-token CE on the sheet only (its text and ABC_END, the sheet's stop) or
        on the music only (codec ids and MUSIC_END; the MUSIC_START after ABC_END is forced when sampling), as the
        mean over those targets. KL, when on, still covers every position. Covers the do_separation stems too, which
        come through here as well. Relies on upstream's cross_entropy keeping ignore_index -100."""
        if self.ar_loss_on == "all":
            return super()._ar_losses(ar_embeds, ids, prefix, total, whole_song)
        n = ids.shape[0]
        # exactly one ABC_END: sheet ids stay below EOD (encode_abc) and codec ids start at CODEC_OFFSET
        k = int((ids == ABC_END).nonzero()[0, 0])
        sheet = self.ar_loss_on == "sheet"
        if sheet and k > self.abc_max_tokens and not self._long_sheet_noted:
            self._long_sheet_noted = True
            print_acc(f"YuE2: a training sheet runs {k} tokens, past abc_max_tokens {self.abc_max_tokens}, where samples stop writing one")
        if sheet and self.ar_kl_weight <= 0:
            # causal AR: no sheet logit sees past ABC_END, so the song is left out (same CE, a fraction of the compute)
            return super()._ar_losses(ar_embeds[:, : ar_embeds.shape[1] - n + k + 1], ids[: k + 1], prefix, total, whole_song)
        pos = torch.arange(n, device=ids.device)
        targets = torch.where(pos <= k if sheet else pos > k + 1, ids, torch.full_like(ids, -100))
        counted = k + 1 if sheet else n - k - 2  # never 0: ABC_END, or at least one codec id
        debug, self.debug = self.debug, False  # upstream's p(target) readout gathers every target and cannot take -100
        try:
            ce, kl, _ = super()._ar_losses(ar_embeds, targets, prefix, total, whole_song)
        finally:
            self.debug = debug
        # upstream divides the summed CE by all n positions
        return ce * (n / counted), kl, f"[ar_loss_on {self.ar_loss_on}: {counted}/{n} targets, sheet {k}]"

    def _get_semantic_tokenizer(self):
        """A bare file name (``WannabeTokenizer.safetensors``) means a file in <MODELS_PATH>/ai_toolkit/, where the
        community heads live too; upstream's resolver would read it as an 'org/repo/file' download."""
        p = str(self.semantic_head_path)
        if not os.path.exists(p) and "/" not in p:
            from toolkit.paths import MODELS_PATH

            local = os.path.join(MODELS_PATH, AI_TOOLKIT_DIR, p)
            if os.path.exists(local):
                self.semantic_head_path = local
        return super()._get_semantic_tokenizer()

    def get_latent_space_version(self):
        """Upstream keys the latent cache by the head file's stem; a Yue2SemanticTokenizer end-to-end pair is always
        ``<run>/<best|last>/head.safetensors``, so its run and tag name the cache instead (every pair's tokens differ)."""
        version = super().get_latent_space_version()
        if os.path.basename(str(self.semantic_head_path)) == "head.safetensors":
            d = os.path.dirname(os.path.abspath(self.semantic_head_path))
            tag = re.sub(r"[^A-Za-z0-9]+", "-", f"{os.path.basename(os.path.dirname(d))}-{os.path.basename(d)}")
            version = (version[: -len("_head")] if version.endswith("_head") else version) + f"_{tag}"
        return version

    # ------------------------------------------------------------------ single-expert training
    def get_loss_target(self, *args, **kwargs):
        target = super().get_loss_target(*args, **kwargs)
        # an AR-only run skips the NAR (see get_noise_prediction): its flow term is pinned to 0
        return torch.zeros_like(target) if self.train_experts == "ar" else target

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, batch=None, **kwargs):
        # `batch` stays a named parameter: predict_noise only forwards it when it sees it in the signature
        if self.train_experts != "ar":
            pred = super().get_noise_prediction(latent_model_input, timestep, text_embeddings, batch=batch, **kwargs)
        else:
            # AR-only: upstream's step still computes the AR losses, but the NAR call answers zeros instead of running
            # the decoder; against get_loss_target's zero target the flow term is exactly 0
            nar = self.model.nar
            nar.forward = lambda x, *args, **kw: torch.zeros_like(x)
            try:
                pred = super().get_noise_prediction(latent_model_input, timestep, text_embeddings, batch=batch, **kwargs)
            finally:
                del nar.forward  # back to the class's own forward
        # exact-token replay: with probability replay_ratio, one track the AR wrote itself joins this step's AR loss
        self._pending_replay_ce = None
        if (self.replay_ratio > 0 and torch.is_grad_enabled() and self._pending_aux_loss is not None
                and random.random() < self.replay_ratio):
            ce, kl = self._replay_ar_loss()
            self._pending_replay_ce = ce
            self._pending_aux_loss = self._pending_aux_loss + ce * self.replay_weight * self.ar_loss_weight
            if kl is not None and self.ar_kl_weight > 0:
                self._pending_aux_loss = self._pending_aux_loss + kl * self.ar_kl_weight
        return pred

    def get_additional_loss(self, pred, target):
        aux = super().get_additional_loss(pred, target)
        ce = self._pending_replay_ce
        self._pending_replay_ce = None
        if aux is not None and ce is not None:
            self.additional_loss_logs["loss/ar_replay_ce"] = ce.detach().float().item()
        return aux

    # ------------------------------------------------------------------ exact-token replay
    def _replay_tracks(self) -> List[str]:
        """Track folders of the replay corpora (minted layout: tracks/<pid>/{semantic.npy, request.json[, score.abc]}),
        scanned once."""
        if self._replay_pool is None:
            pool: List[str] = []
            for root in self.replay_dirs:
                tracks = os.path.join(root, "tracks")
                found = [d for d in sorted(glob.glob(os.path.join(tracks, "*")))
                         if os.path.exists(os.path.join(d, "semantic.npy")) and os.path.exists(os.path.join(d, "request.json"))]
                if not found:
                    raise ValueError(f"model_kwargs.replay_dirs: no tracks with semantic.npy + request.json under {tracks}")
                pool += found
            print_acc(f"YuE2: exact-token replay pool {len(pool)} tracks from {len(self.replay_dirs)} corpora "
                      f"(ratio {self.replay_ratio}, weight {self.replay_weight})")
            self._replay_pool = pool
        return self._replay_pool

    def _minted_track(self, folder: str, abc_dropout: float = 0.0):
        """A track folder in the minted layout -> (codec ids [T] int64, style, lyrics, cot, sheet ids): its codes and the
        prompt they were minted from (the request's style / lyrics / cot). The sheet: abc_tokens.npy (its exact ids) >
        score.abc (re-encoded) > none. Their corpus pull carries no sheet files (item.json's abc_chars only), so those
        tracks are off-mode, as is one whose sheet is dropped (probability ``abc_dropout``)."""
        with open(os.path.join(folder, "request.json"), encoding="utf-8") as f:
            req = json.load(f)
        codes = load_codes(os.path.join(folder, "semantic.npy"))
        cot = str(req.get("cot") or "off")
        ids_path, abc_path = os.path.join(folder, "abc_tokens.npy"), os.path.join(folder, "score.abc")
        has_sheet = os.path.exists(ids_path) or os.path.exists(abc_path)
        if cot != "off" and (not has_sheet or (abc_dropout > 0 and random.random() < abc_dropout)):
            cot = "off"
        abc_ids: List[int] = []
        if cot != "off":
            if os.path.exists(ids_path):
                abc_ids = [int(v) for v in np.load(ids_path)]
            else:
                with open(abc_path, encoding="utf-8") as f:
                    abc_ids = self.tokenizer.encode_abc(f.read())
        return codes, str(req.get("style") or ""), str(req.get("lyrics") or ""), cot, abc_ids

    def _replay_ar_loss(self):
        """Next-token CE (and KL) of one replay track: the AR's own codes under the prompt they were minted from, built
        like a real item. The sheet is dropped with probability abc_dropout, as upstream drops real sheets.
        -> (ce, kl_or_None)."""
        codes, style, lyrics, cot, abc_ids = self._minted_track(random.choice(self._replay_tracks()), self.abc_dropout)
        device = self.device_torch
        tokens = torch.as_tensor(codes, device=device)
        head = self.tokenizer.prefix_head_ids(style, lyrics, cot=cot)
        prefix = self.model.ar.embed(torch.tensor([head], device=device))[0]
        total = tokens.shape[0]
        limit = total if self.ar_max_tokens <= 0 else min(total, self.ar_max_tokens)
        embeds, ids = self._ar_inputs(prefix, torch.tensor(abc_ids, dtype=torch.long, device=device), tokens[:limit], end_token=limit == total)
        ce, kl, _ = self._ar_losses(embeds, ids, prefix, total, limit == total)
        return ce, kl
