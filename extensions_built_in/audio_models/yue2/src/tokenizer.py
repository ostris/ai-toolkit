"""Text side and audio side of YuE2 conditioning.

- ``YuE2TextTokenizer``: the Qwen BPE that ComfyUI embeds in the checkpoint
  (``text_encoders.yue2_tokenizer_json``), plus the prompt/prefix protocol.
- ``SemanticTokenizer``: audio -> codec ids at 25 Hz, from either head format:
  * the community head (Mothersuperior v4..v9): MERT-v2-FullSong layer-20 features
    (the stock encoder from a LOCAL folder, MERT_REPO, through the same ``mert2.py`` port),
    per-track instance normalised, through an 8-layer transformer classifier over
    512-frame windows;
  * a Yue2SemanticTokenizer end-to-end pair (``runs/<run>/best/head.safetensors`` with
    ``head.kind = cvq`` in its metadata): the YuE2 report's form, a causal MERT-v2-30s
    encoder adapted to YuE2's ids (``model.safetensors`` + ``config.json`` beside the head,
    run whole-song through the local ``mert2.py`` port) and a Linear 1024->32 + cosine
    codebook quantizer (wider projection / codebook and an optional SwiGLU block in front,
    sized from the file's tensors); per-frame, label offset from the metadata (frame t -> token t+1).
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from toolkit.basic import UnusableFileError

from .model import ABC_END, ABC_START, CODEC_SIZE, EOD, INSTRUCTIONS, MUSIC_START

# The community heads' encoder: a LOCAL folder holding this repo's config.json + model.safetensors (sha256 e6dd2ab1...28391350d),
# run by src/mert2.py; never fetched at run time and never read from the HF cache (yue2_model.resolve_local_folder finds it). Until
# 2026-11-12 it was transformers remote code + weights pulled through ~/.cache/huggingface by this id.
MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_SAMPLE_RATE = 24000
MERT_LAYER = 20
HEAD_REPO = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
HEAD_FILE = "tokenizer_head_joint_v4.pt"
NAR_LORA_FILE = "nar_lora_joint_v4.pt"
HEAD_WINDOW = 512


class YuE2TextTokenizer:
    def __init__(self, tokenizer_json: bytes):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(unicodedata.normalize("NFC", text)).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def prefix_head_ids(self, style: str, lyrics: str, cot: str = "off") -> List[int]:
        """Prompt up to and including ABC_START (ComfyUI's ``prefix``); the ABC block and
        ``[ABC_END, MUSIC_START]`` follow (see ``abc_tail_ids``)."""
        prompt = f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
        return [EOD] + self.encode(prompt) + [ABC_START]

    @staticmethod
    def abc_tail_ids(abc_ids: List[int]) -> List[int]:
        return list(abc_ids) + [ABC_END, MUSIC_START]

    def encode_abc(self, abc: str) -> List[int]:
        ids = self.encode(abc)
        if any(not 0 <= t < EOD for t in ids):
            raise ValueError("ABC ids must stay inside the ordinary text vocabulary")
        return ids

    def prefix_ids(self, style: str, lyrics: str, cot: str = "off", abc: str = "") -> List[int]:
        """Everything the AR expert sees before the first codec token."""
        abc_ids = [] if cot == "off" else self.encode_abc(abc)
        return self.prefix_head_ids(style, lyrics, cot) + self.abc_tail_ids(abc_ids)

    def negative_prefix_ids(self, cot: str, abc_ids: List[int]) -> List[int]:
        """The reference package's CFG negative (``protocol.negative_prefix``): the bare instruction, no tags or
        lyrics, then for full/melody the exact sheet ids the positive branch sings. Off mode's negative has no ABC
        block at all, unlike its positive prefix."""
        base = [EOD] + self.encode(INSTRUCTIONS[cot])
        if cot == "off":
            return base + [MUSIC_START]
        return base + [ABC_START] + self.abc_tail_ids(abc_ids)


class TokenizerHead(nn.Module):
    def __init__(self, din=1024, d=512, layers=8, heads=8, vocab=CODEC_SIZE, window=HEAD_WINDOW):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, : x.shape[1]])))


class SwiGLU(nn.Module):
    """Pre-norm residual SwiGLU feed-forward, x + down(silu(gate(LN x)) * up(LN x)) (Yue2SemanticTokenizer's head.SwiGLU)."""

    def __init__(self, din, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(din)
        self.gate = nn.Linear(din, hidden, bias=False)
        self.up = nn.Linear(din, hidden, bias=False)
        self.down = nn.Linear(hidden, din, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class CVQHead(nn.Module):
    """The report's quantizer: Linear 1024->dim (32 in the report), cosine against a 32,768 x dim codebook, /tau; with
    ``hidden`` > 0 a SwiGLU block in front of the projection. Per frame; logits [.., vocab]."""

    def __init__(self, din=1024, dim=32, vocab=CODEC_SIZE, tau=0.07, hidden=0):
        super().__init__()
        self.adapter = SwiGLU(din, hidden) if hidden else None
        self.proj = nn.Linear(din, dim)
        self.codebook = nn.Parameter(torch.zeros(vocab, dim))
        self.tau = tau

    def forward(self, x):
        if self.adapter is not None:
            x = self.adapter(x)
        return F.normalize(self.proj(x), dim=-1) @ F.normalize(self.codebook, dim=-1).T / self.tau


def head_metadata(head_path: str) -> dict:
    """The metadata a Yue2SemanticTokenizer checkpoint carries (``head.*``: kind, source_spec, offset, instnorm, tau;
    ``encoder.config`` when the file is a unified tokenizer with its encoder inside); {} for the community files and .pt."""
    if not str(head_path).endswith(".safetensors"):
        return {}
    with safe_open(head_path, "pt") as f:
        return {k: v for k, v in (f.metadata() or {}).items() if k.startswith(("head.", "encoder."))}


def head_kind(head_path: str) -> str:
    """'cvq' (a Yue2SemanticTokenizer pair or unified file: its own encoder) or 'transformer' (the community heads, which carry
    no metadata and read stock MERT-v2-FullSong)."""
    return head_metadata(head_path).get("head.kind", "transformer")


def _rebuild_rotary(model: nn.Module) -> int:
    """MERT2's RotaryEmbedding keeps ``inv_freq`` as a non-persistent buffer; transformers'
    meta-device loading materializes it uninitialized (zeros on a fresh GPU, garbage or NaN
    otherwise). Recompute it from the module's own formula and drop the cos/sin cache."""
    n = 0
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "head_dim") and hasattr(m, "base"):
            inv = 1.0 / (m.base ** (torch.arange(0, m.head_dim, 2, dtype=torch.float32) / m.head_dim))
            m.inv_freq = inv.to(device=m.inv_freq.device)
            for attr, val in (("_cos", None), ("_sin", None), ("_sequence_length", 0), ("_cache_device", None)):
                if hasattr(m, attr):
                    setattr(m, attr, val)
            n += 1
    return n


SHEETSAGE_REPO = "m-a-p/SheetSage2"
# SheetSage2's config.json + inference code (no weights: SHEETSAGE_FILE) are a LOCAL folder holding this repo at this revision, never
# fetched at run time and never read from the HF cache (yue2_model.resolve_local_folder finds the folder). The code is executed, so
# it is this reviewed pin: 55bfe14 = 2026-09-24 "Fix chord spelling for non-C candidate roots" (diff reviewed); earlier code wrote
# sharp-side names for chromatic chords (A# for Bb in G major, G# for Ab in C minor). Until 2026-11-11 it was transformers remote
# code pulled through ~/.cache/huggingface, and transcription died offline once that cache was emptied.
SHEETSAGE_REVISION = "55bfe14e32d8b663629b3a86b0f3285c2ca5da0b"
SHEETSAGE_FILE = "Comfy-Org/YuE2/audio_encoders/sheetsage2_bf16.safetensors"
# tied to token_embedding.weight (loaded); everything else in the Comfy file maps 1:1 onto upstream
NONPERSISTENT_OK = {"decoder.embed_tokens.weight"}
SHEETSAGE_SAMPLE_RATE = 24000
# SheetSage2-Refined (anime-song): its code is vendored in ../sheetsage2_refined (MIT, 0.5.2); its weights (CC BY-NC) are this repo at
# this revision, the 0.5.2 release the code matches (the weights transcription loads are byte-identical back to 0.5.0, whose code
# made our earlier Refined sheets: 0.5.1 / 0.5.2 changed only the decoder-free key and beat trackers). Its encoder is SheetSage2's,
# rebuilt from SHEETSAGE_FILE (refined_base_weights).
REFINED_REPO = "anime-song/SheetSage2-Refined"
REFINED_REVISION = "5e9c494941b8c0bb322d9f17f365a6f886d5e562"


def code_package(folder: str, name: str) -> str:
    """Make ``folder`` (plain .py modules importing each other relatively; no __init__.py needed) importable as the package ``name``
    -> name. One folder per name for the life of the process: a second folder under the same name raises."""
    import importlib.machinery
    import importlib.util
    import sys

    folder = os.path.abspath(folder)
    if name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [folder]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    elif list(sys.modules[name].__path__) != [folder]:
        raise ValueError(f"package {name!r} is already loaded from {list(sys.modules[name].__path__)}, not {folder}")
    return name


class SheetSage2Transcriber:
    """Audio -> ABC lead sheet with the released SheetSage2 (its own inference code, from a local folder), the same
    transcriber ComfyUI ships for cover mode. ``full`` keeps chords, ``melody`` drops them."""

    def __init__(self, weights_path: str, code_dir: str):
        """``weights_path``: the Comfy repack ``audio_encoders/sheetsage2_bf16.safetensors`` (merged
        weights); ``code_dir``: a LOCAL folder with m-a-p/SheetSage2's config.json + inference code (SHEETSAGE_REPO at
        SHEETSAGE_REVISION), imported from there as a package. Never the hub, and nothing of ~/.cache/huggingface."""
        from .shims import install_shims

        install_shims()  # mir_eval.chord / pretty_midi stand-ins unless the real packages exist
        import importlib
        import sys

        from safetensors.torch import load_file

        from .sheetsage_decoder import ScoreDecoder
        from .sheetsage_generate import FastConstrainedGenerate

        if not os.path.isfile(os.path.join(code_dir, "config.json")):
            raise FileNotFoundError(f"SheetSage2 code folder {code_dir!r} is not a local folder with config.json")
        modeling = importlib.import_module(code_package(code_dir, "sheetsage2_code") + ".modeling_sheetsage2")
        cfg = modeling.SheetSage2Config.from_pretrained(code_dir, local_files_only=True)
        cfg.weights_format = "merged"  # the Comfy file carries the LoRA already folded into the encoder
        cls = modeling.SheetSage2Model
        sys.modules[cls.__module__].BartDecoder = ScoreDecoder  # upstream targets transformers 4.45's BartDecoder
        pipeline = importlib.import_module(cls.__module__.rsplit(".", 1)[0] + ".pipeline_sheetsage2")
        if not isinstance(pipeline.constrained_prompt_generate, FastConstrainedGenerate):
            # per-window decode loop: static cache + CUDA graph instead of upstream's per-token Python loop
            pipeline.constrained_prompt_generate = FastConstrainedGenerate(pipeline.constrained_prompt_generate)
        cls.post_init = lambda self: None  # transformers-5 tied-weight bookkeeping rejects upstream's list; weights load explicitly
        model = cls(cfg)
        sd = {k: v.to(torch.float32) for k, v in load_file(weights_path).items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if k not in NONPERSISTENT_OK]
        if missing or unexpected:
            raise ValueError(f"SheetSage2 weights do not match: missing {missing[:5]} unexpected {unexpected[:5]}")
        model.output_projection.weight = model.token_embedding.weight  # keep the tie after loading
        self.model = model.eval()
        self.model.requires_grad_(False)
        _rebuild_rotary(self.model)
        self.warnings: List[str] = []  # repair notes for the caller to print with the track path

    def to(self, device):
        self.model.to(device)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor, sample_rate: int, cot: str = "full") -> str:
        """waveform [C, samples] -> ABC text (``cot`` "full" or "melody")."""
        mono = waveform.float().mean(0) if waveform.dim() == 2 else waveform.float()
        result = self.model.transcribe(mono.cpu(), sampling_rate=sample_rate, melody_only=cot == "melody")
        return sheet_abc(self.model, result, cot, self.warnings)


def sheet_abc(owner, result: dict, cot: str, warnings: List[str], who: str = "SheetSage2") -> str:
    """The ABC of an exported transcription (SheetSage2's transcribe() dict, or SheetSage2Exporter.export's equivalent). A sheet the
    strict export rejected is rebuilt with the bad rows repaired (sheetsage_repair; ``owner`` = any object of SheetSage2's code
    package, which names the modules it rebuilds with), else the cacher drops the track. Repair notes go to ``warnings``."""
    from .sheetsage_repair import AbcRepairError, repair_abc

    abc = result.get("abc") or ""
    if not abc.strip():
        error = result.get("abc_error") or "unknown reason"
        try:
            abc, summary = repair_abc(owner, result, melody_only=cot == "melody")
        except AbcRepairError as exc:
            raise UnusableFileError(f"{who} produced no ABC: {error}; repair failed: {exc}") from exc
        warnings.append(f"{who} ABC repaired ({error}): {summary}")
    return abc


class SheetSage2Exporter:
    """SheetSage2's own ABC writer without its model: SheetSage2's event stream (another decoder's: SheetSage2-Refined's) -> the ABC
    dialect the AR was trained on, through ``exports_sheetsage2.export_result`` and its tokenizer, from the same local code folder
    SheetSage2Transcriber reads (``code_dir``). No weights load."""

    def __init__(self, code_dir: str):
        from .shims import install_shims

        install_shims()
        import importlib

        if not os.path.isfile(os.path.join(code_dir, "config.json")):
            raise FileNotFoundError(f"SheetSage2 code folder {code_dir!r} is not a local folder with config.json")
        package = code_package(code_dir, "sheetsage2_code")
        configuration = importlib.import_module(package + ".configuration_sheetsage2")
        tokenization = importlib.import_module(package + ".tokenization_sheetsage2")
        self.exports = importlib.import_module(package + ".exports_sheetsage2")
        cfg = configuration.SheetSage2Config.from_pretrained(code_dir, local_files_only=True)
        # the tokenizer SheetSage2Model builds from the same config
        self.tokenizer = tokenization.SheetSage2Tokenizer(
            cfg.input_audio_length, cfg.time_hz, cfg.tokenizer_schema_version, expected_fingerprint=cfg.tokenizer_fingerprint,
        )

    def export(self, events: list, duration: float, cot: str, warnings: List[str], who: str) -> str:
        exported = self.exports.export_result({"events": events}, self.tokenizer, duration=duration, melody_only=cot == "melody")
        payload = exported.pop("payload")
        return sheet_abc(self.tokenizer, {**exported, **payload, "events": events, "duration_seconds": duration}, cot, warnings, who)


def _refined_encoder_name(k: str) -> str:
    """SheetSage2's MERT2 encoder tensor name (``encoder.`` stripped) -> SheetSage2-Refined's"""
    front = {"feature_extractor.spectrogram.window": "frontend.window", "feature_extractor.mel_scale.fb": "frontend.mel_fb",
             "feature_extractor.mel_mean": "frontend.mel_mean", "feature_extractor.mel_std": "frontend.mel_std"}
    if k in front:
        return front[k]
    parts = k.split(".")
    if parts[0] == "subsampling_module":        # subsampling_module.<stage>.(resampling_layer.<i> | convnext_layers.<n>.<block>.<i>).<p>
        stage = parts[1]
        if parts[2] == "resampling_layer":
            return f"stages.{stage}." + {"0": "down_norm", "2": "down_conv"}[parts[3]] + "." + parts[4]
        n, block, i, p = parts[3], parts[4], parts[5], parts[6]
        if block == "depthwise_block":          # Sequential(Transpose, Conv1d, Transpose): index 1 = the conv
            return f"stages.{stage}.layers.{n}.depthwise.{p}"
        name = {"0": "norm", "1": "fc1", "3": "grn", "4": "fc2"}[i]   # pointwise_block: LayerNorm, Linear, GELU, GRN, Linear
        if name == "grn":                       # GRN weight / bias [1, 1, 4d] -> grn_gamma / grn_beta [4d]
            return f"stages.{stage}.layers.{n}.grn_{'gamma' if p == 'weight' else 'beta'}"
        return f"stages.{stage}.layers.{n}.{name}.{p}"
    if parts[0] == "layers":
        table = {"ffn1_layer_norm": "ffn1_norm", "ffn2_layer_norm": "ffn2_norm", "attn_layer_norm": "attn_norm", "final_layer_norm": "final_norm",
                 "ffn1.w_1": "ffn1.fc1", "ffn1.w_2": "ffn1.fc2", "ffn2.w_1": "ffn2.fc1", "ffn2.w_2": "ffn2.fc2",
                 "attn.query_proj": "attn.q", "attn.key_proj": "attn.k", "attn.value_proj": "attn.v", "attn.out_proj": "attn.out",
                 "conv_module.layer_norm": "conv.norm", "conv_module.conv_block.1": "conv.pointwise_in", "conv_module.conv_block.3": "conv.depthwise",
                 "conv_module.conv_block.4.1": "conv.depthwise_norm", "conv_module.conv_block.6": "conv.pointwise_out"}
        head, p = ".".join(parts[2:]).rsplit(".", 1)
        return f"layers.{parts[1]}.{table[head]}.{p}"
    raise KeyError(k)


def _refined_decoder_name(k: str) -> str:
    """SheetSage2 tensor name outside ``encoder.`` -> SheetSage2-Refined's; '' = not carried (the output projection: Refined ties it)"""
    top = {"layer_weight": "layer_weights", "encoder_projection.weight": "encoder_projection.weight", "encoder_projection.bias": "encoder_projection.bias",
           "token_embedding.weight": "decoder.token_embedding.weight", "decoder.embed_positions.weight": "decoder.position_embedding.weight",
           "decoder.layernorm_embedding.weight": "decoder.embedding_norm.weight", "decoder.layernorm_embedding.bias": "decoder.embedding_norm.bias",
           "output_projection.weight": ""}
    if k in top:
        return top[k]
    parts = k.split(".")                        # decoder.layers.N.<module>.<p> | decoder.layers.N.<attn>.<proj>.<p>
    if parts[:2] != ["decoder", "layers"]:
        raise KeyError(k)
    mod = {"self_attn": "self_attn", "encoder_attn": "cross_attn", "self_attn_layer_norm": "self_attn_norm", "encoder_attn_layer_norm": "cross_attn_norm",
           "final_layer_norm": "final_norm", "fc1": "fc1", "fc2": "fc2"}[parts[3]]
    if parts[3] in ("self_attn", "encoder_attn"):
        return f"decoder.layers.{parts[2]}.{mod}." + {"q_proj": "q", "k_proj": "k", "v_proj": "v", "out_proj": "out"}[parts[4]] + f".{parts[5]}"
    return f"decoder.layers.{parts[2]}.{mod}.{parts[4]}"


def refined_base_weights(sheetsage_file: str, out_path: str):
    """SheetSage2's weights in SheetSage2-Refined's tensor layout, written once to ``out_path`` (its model folder's ``base_weights``)
    from the Comfy file SheetSage2Transcriber already loads (``sheetsage_file``): the same tensors under Refined's module names, so
    its 2.7 GB repack is never downloaded. Every name and shape is checked against Refined's own model before the file is written."""
    from safetensors.torch import load_file, save_file

    from ..sheetsage2_refined.model import SheetSage

    src = load_file(sheetsage_file)
    tensors = {}
    for k, v in src.items():
        if k.startswith("encoder."):
            name = "encoder." + _refined_encoder_name(k[len("encoder."):])
            tensors[name] = v.reshape(-1) if name.endswith(("grn_gamma", "grn_beta")) else v
        elif _refined_decoder_name(k):
            tensors[_refined_decoder_name(k)] = v
    want = {k: tuple(v.shape) for k, v in SheetSage().state_dict().items()}
    missing, extra = sorted(set(want) - set(tensors)), sorted(set(tensors) - set(want))
    # the token embedding is padded to Refined's vocabulary by its own loader
    bad = [k for k in want.keys() & tensors.keys() if k != "decoder.token_embedding.weight" and tuple(tensors[k].shape) != want[k]]
    if missing or extra or bad:
        raise ValueError(f"{sheetsage_file} does not map onto SheetSage2-Refined: missing {missing[:5]} extra {extra[:5]} shape {bad[:5]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, out_path + ".tmp",
              metadata={"source": os.path.basename(sheetsage_file), "layout": "sheetsage2_refined"})
    os.replace(out_path + ".tmp", out_path)


def refined_weight_files(config: dict) -> List[str]:
    """The files of a SheetSage2-Refined model folder that transcription loads, besides ``base_weights``: the decoder, the bar
    features and every rescoring head (not the decoder-free trackers)."""
    bar = config.get("bar_features") or {}
    names = [config.get("decoder"), bar.get("projection"), *bar.get("heads", [])]
    for spec in config.get("rescoring", {}).values():
        names += spec["head"] if isinstance(spec["head"], list) else [spec["head"]]
    return sorted({n for n in names if n})


def _voices(abc: str) -> Tuple[Dict[str, str], List[str]]:
    """The score's voices (id -> its attributes: clef, name=..., from every V: line) and the ids named like "vocal"."""
    declared = {}
    for line in abc.splitlines():
        m = re.match(r"\s*V:\s*(\S+)(.*)", line)
        if m:
            declared[m[1]] = declared.get(m[1], "") + m[2]
    return declared, [v for v, attrs in declared.items() if "vocal" in (v + attrs).lower()]


# in a music line: a note's pitch (accidental, letter, octave marks), or a span whose letters are not pitches and is kept
# whole: a quoted chord symbol or annotation, an inline field, a decoration, a trailing comment
_PITCH = re.compile(r'"[^"]*"|\[[A-Za-z]:[^\]]*\]|![^!]*!|\+[^+\s]*\+|%.*|(?P<acc>\^\^|\^|__|_|=)?(?P<letter>[A-Ga-g])(?P<marks>[,\']*)')


def shift_vocal_octaves(abc: str, octaves: int) -> Tuple[str, Dict]:
    """The score with every voice named like "vocal" moved by ``octaves`` octaves: each note's letter case and , / ' marks change and
    nothing else, so accidentals, lengths, ties, chord symbols, inline fields, header and field lines and every other voice stay byte
    for byte; grace notes move with their voice. -> (abc, {"voices": moved voice ids, "notes": notes moved})."""
    _, vocal = _voices(abc)
    moved = 0

    def move(m):
        nonlocal moved
        if not m["letter"]:
            return m[0]
        moved += 1
        octave = (4 if m["letter"].isupper() else 5) + m["marks"].count("'") - m["marks"].count(",") + octaves   # C = C4, c = C5
        return (m["acc"] or "") + (m["letter"].upper() + "," * (4 - octave) if octave < 5 else m["letter"].lower() + "'" * (octave - 5))

    out, cur = [], None
    for line in abc.splitlines():
        s = line.strip()
        field = re.match(r"([A-Za-z]):(.*)", s)
        if field and field[1] == "V":
            cur = field[2].split()[0] if field[2].strip() else None
        out.append(line if not s or s.startswith("%") or field or cur not in vocal else _PITCH.sub(move, line))
    return "\n".join(out) + ("\n" if abc.endswith("\n") else ""), {"voices": vocal, "notes": moved}


def sheetsage2_chord(label: str, chords) -> str:
    """A SheetSage2-Refined chord label (mir_eval syntax over its 62 qualities and any bass, e.g. ``F:maj7(9)``) -> the nearest label in
    SheetSage2's own chord vocabulary (``chords``: the vendored vocab.FULL_CHORDS), the only ones SheetSage2's exporter can spell and the
    dialect YuE2's sheets were written in. Tensions in parentheses are dropped (maj7(9) -> maj7; a sus4 keeps its b7, which SheetSage2
    has as sus4(b7)), a power chord reads as maj, and a bass SheetSage2 has no inversion for reads as root position."""
    if label in chords or label in ("N", "X"):
        return label
    root, _, rest = label.partition(":")
    body, _, bass = rest.partition("/")
    shorthand, _, tensions = body.partition("(")
    quality = {"5": "maj"}.get(shorthand, shorthand)
    if quality == "sus4" and "b7" in tensions.rstrip(")").split(","):
        quality = "sus4(b7)"
    for candidate in ((f"{root}:{quality}/{bass}",) if bass else ()) + (f"{root}:{quality}",):
        if candidate in chords:
            return candidate
    raise ValueError(f"SheetSage2-Refined chord {label!r} has no counterpart in SheetSage2's chord vocabulary")


def fit_bar_meters(events: list) -> list:
    """SheetSage2-Refined counts each bar's beats from its own downbeats (eighth position 0) but can leave the declared meter too short for
    them: a (3, 8) meter over bars of six eighths, or a 4/4 bar with a fifth beat. SheetSage2's exporter rejects every such row. A bar whose
    positions do not fit its meter gets a meter that holds them (same denominator, numerator = its last beat + 1), and the declared meter
    is restored on the next downbeat; bars that fit are left as they are. Rows off the denominator's grid, and bars whose meter changes
    inside them, are left for sheet_abc's repair. Edits ``events`` in place (a fresh transcription) and returns it."""
    from fractions import Fraction

    bars, meter = [], None                      # per bar: (meter in effect, rhythm) for each beat row, as the exporter reads them
    for event in events:
        rhythm = event["values"].get("rhythm") or {}
        meter = tuple(rhythm["meter"]) if "meter" in rhythm else meter
        if rhythm.get("eighth_position") is None or meter is None:
            continue
        if rhythm["eighth_position"] == 0 or not bars:
            bars.append([])
        bars[-1].append((meter, rhythm))
    widened = False
    for bar in bars:
        (num, den), first = bar[0]
        if widened and "meter" not in first:
            first["meter"] = (num, den)         # the meter in effect again after a widened bar
        widened = False
        if len({m for m, _ in bar}) > 1:
            continue
        beats = [Fraction(int(rhythm["eighth_position"]) * den, 8) for _, rhythm in bar]
        if any(b.denominator != 1 for b in beats) or all(0 <= b < num for b in beats):
            continue
        first["meter"], widened = (int(max(beats)) + 1, den), True
    return events


class RefinedTranscriber:
    """Audio -> ABC lead sheet with SheetSage2-Refined (anime-song; ../sheetsage2_refined): its fine-tuned decoder and rescoring heads
    over SheetSage2's encoder decode SheetSage2's event stream, and SheetSage2's own exporter writes it (SheetSage2Exporter), after the
    stream is fitted to what that exporter accepts (sheetsage2_chord, fit_bar_meters). Refined writes the vocal line in the octave sung,
    one below SheetSage2's sheets and YuE2's own, so it is moved up one octave. The same interface as SheetSage2Transcriber (transcribe,
    to, device, warnings)."""

    def __init__(self, folder: str, sheetsage_file: str, code_dir: str):
        """``folder``: a SheetSage2-Refined model folder (config.json + refined_weight_files); its ``base_weights`` is built from
        ``sheetsage_file`` when missing (refined_base_weights). ``code_dir``: SheetSage2's local code folder, for the exporter."""
        from ..sheetsage2_refined.pipeline import SheetSage2Refined
        from ..sheetsage2_refined.vocab import FULL_CHORDS

        self.chords = frozenset(FULL_CHORDS)
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            base = os.path.join(folder, json.load(f)["base_weights"])
        if not os.path.isfile(base):
            print(f"SheetSage2-Refined: writing its encoder weights from {os.path.basename(sheetsage_file)} -> {base}")
            refined_base_weights(sheetsage_file, base)
        self.exporter = SheetSage2Exporter(code_dir)
        self.refined = SheetSage2Refined.load(folder, "cpu")
        self.warnings: List[str] = []

    def to(self, device):
        self.refined.model.to(device)                    # the bar features are a submodule of the model
        for head, _ in self.refined.rescoring.values():  # the rescoring heads are not (shared ones move once, harmlessly twice)
            head.to(device)
        return self

    @property
    def device(self):
        return next(self.refined.model.parameters()).device

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor, sample_rate: int, cot: str = "full") -> str:
        """waveform [C, samples] -> ABC text (``cot`` "full" or "melody"), vocal line in YuE2's sheet octave."""
        import torchaudio

        rate = self.refined.model.sample_rate
        mono = waveform.float().mean(0) if waveform.dim() == 2 else waveform.float()
        mono = torchaudio.functional.resample(mono, sample_rate, rate) if sample_rate != rate else mono
        events = fit_bar_meters(self.refined.transcribe([mono.cpu()])[0])
        for event in events:
            if "chord" in event["values"]:
                event["values"]["chord"] = sheetsage2_chord(event["values"]["chord"], self.chords)
        abc = self.exporter.export(events, len(mono) / rate, cot, self.warnings, who="SheetSage2-Refined")
        return shift_vocal_octaves(abc, 1)[0]


def load_head_state_dict(head_path: str) -> dict:
    """Head weights from either release format of the community tokenizer repo: the original ``.pt``
    (``{"model": state_dict, "cfg": ...}``) or the ``.safetensors`` files, which hold the same state dict
    flat. The v5 / v8 / v9 joint heads (same architecture as v4) are published as safetensors only."""
    if str(head_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(head_path, device="cpu")
        # a unified tokenizer file carries its encoder under encoder.* and the head under head.*; the community head's
        # own final Linear is ALSO named head.* (head.weight / head.bias), so the encoder tensors are the tell, never the prefix
        if any(k.startswith("encoder.") for k in sd):
            sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
        return sd
    return torch.load(head_path, map_location="cpu", weights_only=False)["model"]


def shift_samples(x: torch.Tensor, ms: float, rate: int) -> torch.Tensor:
    """[..., S] advanced (ms > 0: frame t then starts ms later in the song) or delayed (ms < 0) by whole samples, zero-padded, the
    length kept: one sub-frame reading of a song for phase averaging (Yue2SemanticTokenizer's engine.shift_samples, kept identical)."""
    d = int(round(ms * rate / 1000))
    if d > 0:
        return torch.cat([x[..., d:], x.new_zeros(*x.shape[:-1], d)], -1)
    if d < 0:
        return torch.cat([x.new_zeros(*x.shape[:-1], -d), x[..., :d]], -1)
    return x


class SemanticTokenizer(nn.Module):
    """Waveform -> per-frame codec ids (0..32767) at 25 Hz.

    ``phase_tta`` (ms, 2026-11-03): (0,) = one reading; several = phase averaging: the ids of a real song hang on where the 40-ms frame
    grid falls (2 ms of shift changed ~20 % of them, and two files of one song, 6 ms apart on the grid, read different words), so the song
    is read at each sub-frame shift and the head's softmax averaged per frame before the argmax (Yue2SemanticTokenizer's engine.tokenize,
    kept identical)."""

    def __init__(self, head_path: str, mert_dir: Optional[str] = None, debug: bool = False, phase_tta=(0,)):
        """``mert_dir``: the LOCAL folder with MERT-v2-FullSong's config.json + model.safetensors (MERT_REPO), which a community
        (transformer) head reads; a cvq file names or carries its own encoder and needs none. Never a hub id: nothing here
        downloads or reads ~/.cache/huggingface (yue2_model.load_semantic_tokenizer finds the folder for the heads that need it)."""
        super().__init__()
        from .mert2 import MERT2Encoder

        self.debug = debug
        self.phase_tta = tuple(phase_tta)
        meta = head_metadata(head_path)
        self.kind = head_kind(head_path)
        # what the head reads: the community head normalises each track's features, the cvq pairs read them raw;
        # a cvq pair's frame t predicts token t + offset, so its stream is shifted onto the token clock
        self.instnorm = meta.get("head.instnorm", "1") == "1"
        self.offset = int(meta.get("head.offset", 0))
        sd = load_head_state_dict(head_path)
        # the encoder the head reads, the block it taps and how a song is cut (mert_features): a cvq pair's own adapted encoder at
        # its spec's tap, whole song when chunk is 0; the community head's stock MERT-v2-FullSong, block 20, in 30-s pieces
        if self.kind == "cvq":
            spec = json.loads(meta["head.source_spec"])
            # a unified file carries its encoder; a pair's head has it beside it (or at the spec's path)
            encoder = head_path if "encoder.config" in meta else self._encoder_dir(head_path, spec)
            causal, self.layer, self.chunk = bool(spec["causal"]), int(spec["layer"]), float(spec["chunk"])
            self.head = CVQHead(dim=sd["codebook"].shape[1], tau=float(meta.get("head.tau", 0.07)),
                                hidden=sd["adapter.up.weight"].shape[0] if "adapter.up.weight" in sd else 0)
        elif self.kind == "transformer":
            if not (mert_dir and os.path.isdir(mert_dir)):
                raise FileNotFoundError(f"community head {head_path}: its encoder ({MERT_REPO}) must be a local folder, got {mert_dir!r}")
            encoder, causal, self.layer, self.chunk = mert_dir, False, MERT_LAYER, 30.0
            self.head = TokenizerHead()
        else:
            raise ValueError(f"unknown semantic head kind {self.kind!r} in {head_path}")
        self.encoder = MERT2Encoder.from_pretrained(encoder, causal=causal, layers=self.layer + 1)   # frozen, in eval mode
        self.head.load_state_dict(sd)
        self.head.eval().requires_grad_(False)

    @staticmethod
    def _encoder_dir(head_path: str, spec: dict) -> str:
        """The adapted encoder of a cvq pair: the spec's path when it exists as given, else the folder the head sits in
        (a pair is saved as one folder: head.safetensors + model.safetensors + config.json)."""
        mert = spec["mert"]
        if os.path.isdir(mert) and os.path.exists(os.path.join(mert, "model.safetensors")):
            return mert
        beside = os.path.dirname(os.path.abspath(head_path))
        if os.path.exists(os.path.join(beside, "model.safetensors")) and os.path.exists(os.path.join(beside, "config.json")):
            return beside
        raise FileNotFoundError(f"cvq head {head_path}: its encoder ({mert!r}) is neither at that path nor beside the head")

    @property
    def device(self):
        return next(self.head.parameters()).device

    @torch.no_grad()
    def mert_features(self, mono24: torch.Tensor) -> torch.Tensor:
        """mono24 [samples] at 24 kHz -> [T25, 1024] fp32 features of the head's source: the community head's
        layer 20 of MERT-v2-FullSong in 30-s pieces, or the cvq pair's tap of its adapted causal encoder, whole song."""
        return self.encoder.features(mono24.to(self.device), [self.layer], chunk=self.chunk)[self.layer]

    @torch.no_grad()
    def tokens_from_features(self, feats: torch.Tensor, return_conf: bool = False):
        """[T, 1024] -> [T] long. Windowed inference with edge trimming for the transformer head; one per-frame pass
        for a cvq head, shifted by its label offset onto the token clock (edges repeat the nearest prediction).
        ``return_conf`` also returns the head's probability of each chosen id [T] (diagnostics: how sure the argmax is)."""
        x = feats.float()
        if self.instnorm:
            x = (x - x.mean(0)) / (x.std(0) + 1e-5)
        total = x.shape[0]
        device = self.device
        if self.kind == "cvq":
            out = torch.zeros(total, dtype=torch.long, device=x.device)
            conf = torch.zeros(total, dtype=torch.float32, device=x.device)
            for s0 in range(0, total, 4096):  # [4096, 32768] fp32 logits at a time
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = self.head(x[s0 : s0 + 4096].to(device)).float()
                out[s0 : s0 + 4096] = logits.argmax(-1).to(out.device)
                if return_conf:
                    conf[s0 : s0 + 4096] = logits.softmax(-1).amax(-1).to(conf.device)
            idx = (torch.arange(total, device=x.device) - self.offset).clamp(0, total - 1)
            return (out[idx], conf[idx]) if return_conf else out[idx]
        win = HEAD_WINDOW
        out = torch.zeros(total, dtype=torch.long, device=x.device)
        conf = torch.zeros(total, dtype=torch.float32, device=x.device)
        starts = list(range(0, max(1, total - win + 1), win // 2))
        if starts[-1] + win < total:
            starts.append(max(0, total - win))
        for s0 in starts:
            xw = x[s0 : s0 + win]
            n = xw.shape[0]
            if n < win:
                xw = F.pad(xw, (0, 0, 0, win - n))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = self.head(xw[None].to(device))[0, :n].float()
            pred = logits.argmax(-1)
            lo = s0 + (0 if s0 == 0 else win // 4)
            hi = s0 + n - (0 if s0 + n >= total else win // 4)
            out[lo:hi] = pred[lo - s0 : hi - s0].to(out.device)
            if return_conf:
                conf[lo:hi] = logits.softmax(-1).amax(-1)[lo - s0 : hi - s0].to(conf.device)
        return (out, conf) if return_conf else out

    @torch.no_grad()
    def frame_probs(self, feats: torch.Tensor) -> torch.Tensor:
        """[T, 1024] -> the head's softmax per FEATURE frame [T, V] fp32 on the head's device: tokens_from_features' windows, before
        the label offset (what phase averaging sums over its readings)."""
        x = feats.float()
        if self.instnorm:
            x = (x - x.mean(0)) / (x.std(0) + 1e-5)
        total = x.shape[0]
        device = self.device
        probs = torch.empty(total, CODEC_SIZE, dtype=torch.float32, device=device)
        if self.kind == "cvq":
            for s0 in range(0, total, 4096):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = self.head(x[s0 : s0 + 4096].to(device)).float()
                probs[s0 : s0 + 4096] = logits.softmax(-1)
            return probs
        win = HEAD_WINDOW
        starts = list(range(0, max(1, total - win + 1), win // 2))
        if starts[-1] + win < total:
            starts.append(max(0, total - win))
        for s0 in starts:
            xw = x[s0 : s0 + win]
            n = xw.shape[0]
            if n < win:
                xw = F.pad(xw, (0, 0, 0, win - n))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = self.head(xw[None].to(device))[0, :n].float()
            lo = s0 + (0 if s0 == 0 else win // 4)
            hi = s0 + n - (0 if s0 + n >= total else win // 4)
            probs[lo:hi] = logits.softmax(-1)[lo - s0 : hi - s0]
        return probs

    @torch.no_grad()
    def tokens_averaged(self, mono24: torch.Tensor, return_conf: bool = False):
        """Phase averaging: mono24 [samples] at 24 kHz read at each of ``self.phase_tta``'s sub-frame shifts, the head's softmax averaged
        per frame, the argmax (+ the averaged probability of the chosen id) moved onto the token clock as tokens_from_features does."""
        acc = None
        for ms in self.phase_tta:
            p = self.frame_probs(self.mert_features(shift_samples(mono24, ms, MERT_SAMPLE_RATE)))
            if acc is None:
                acc = p
            else:
                n = min(len(acc), len(p))
                acc = acc[:n].add_(p[:n])
            del p
        conf, out = (acc / len(self.phase_tta)).max(-1)
        del acc
        if self.kind == "cvq":
            idx = (torch.arange(len(out), device=out.device) - self.offset).clamp(0, len(out) - 1)
            out, conf = out[idx], conf[idx]
        return (out, conf) if return_conf else out

    @torch.no_grad()
    def tokenize(self, waveform: torch.Tensor, sample_rate: int, return_conf: bool = False):
        """waveform [C, samples] -> codec ids [T] (long, no vocab offset); with ``return_conf``
        also the head's per-frame probability of the chosen id. Phase averaging over ``self.phase_tta`` unless it is (0,)."""
        import torchaudio

        mono = waveform.float().mean(0) if waveform.dim() == 2 else waveform.float()
        if sample_rate != MERT_SAMPLE_RATE:
            mono = torchaudio.functional.resample(mono, sample_rate, MERT_SAMPLE_RATE)
        if self.phase_tta != (0,):
            result = self.tokens_averaged(mono.to(self.device), return_conf)
            feats = None
        else:
            feats = self.mert_features(mono.to(self.device))
            result = self.tokens_from_features(feats, return_conf)
        out = result[0] if return_conf else result
        if self.debug and out.unique().numel() < 8:
            nan = torch.isnan(feats).sum().item() if feats is not None else "n/a (averaged)"
            print(f"YuE2 tokenizer: degenerate token stream (unique {out.unique().numel()}); MERT features nan {nan}")
        return result
