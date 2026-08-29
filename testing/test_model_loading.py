"""Per-arch model loading + inference smoke test (toolkit/models/v2/PLANNING.md).

Loads one registered arch through its normal loading path (the same
get_model_class -> ModelClass(...).load_model() flow training uses), runs one
small sample generation, and asserts an output file was produced.

Usage:
  python testing/test_model_loading.py --arch zimage          # one arch, in-process
  python testing/test_model_loading.py --all                  # every registered arch,
                                                              # one subprocess each (full
                                                              # unload between archs)
  --allow-download   permit hub downloads (default: HF_HUB_OFFLINE=1, so archs
                     whose weights are not local/cached report SKIP)
  --list             list registered archs
  --device cuda:0

Add a new model type by adding an entry to MODEL_TESTS.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLKIT_ROOT)

from dotenv import load_dotenv

# repo .env carries HF_TOKEN / HF_HOME / MODELS_PATH etc., same as run.py
load_dotenv(os.path.join(TOOLKIT_ROOT, ".env"))

OUTPUT_ROOT = os.path.join(TOOLKIT_ROOT, "testing", ".model_test_outputs")

# arch -> {"model": ModelConfig kwargs, "sample": GenerateImageConfig kwargs}
# Keep samples tiny: this asserts the load/encode/denoise/decode/save path
# works, not quality.
IMG = {"width": 512, "height": 512, "num_inference_steps": 8, "seed": 42}
VID = {"width": 256, "height": 256, "num_inference_steps": 6, "seed": 42, "num_frames": 9}

MODEL_TESTS = {
    "zimage": {
        "model": {"name_or_path": "Tongyi-MAI/Z-Image-Turbo"},
        "sample": {**IMG, "guidance_scale": 1.0},
    },
    "qwen_image": {
        # 20B: quantize to fit a 32GB card
        "model": {"name_or_path": "Qwen/Qwen-Image", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 20, "guidance_scale": 4.0},
    },
    "krea2": {
        "model": {"name_or_path": "krea/Krea-2-Turbo", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "guidance_scale": 1.0},
    },
    "boogu_image": {
        # native ~1024; 512/low-step/high-CFG degenerates to a black frame
        "model": {"name_or_path": "Boogu/Boogu-Image-0.1-Base", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "ernie_image": {
        "model": {"name_or_path": "baidu/ERNIE-Image", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "mageflow": {
        "model": {"name_or_path": "microsoft/Mage-Flow-Base", "quantize": True, "quantize_te": True},
        "sample": IMG,
    },
    "ideogram4": {
        "model": {"name_or_path": "ideogram-ai/ideogram-4-fp8", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "hidream_o1": {
        "model": {"name_or_path": "HiDream-ai/HiDream-O1-Image", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 28, "guidance_scale": 5.0, "seed": 42},
    },
    "anima": {
        "model": {"name_or_path": "circlestone-labs/Anima-Base-v1.0-Diffusers"},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 4.5, "seed": 42},
    },
    "wan21": {
        "model": {"name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
        "sample": {"width": 480, "height": 480, "num_inference_steps": 20, "guidance_scale": 5.0, "seed": 42, "num_frames": 17},
    },
    "wan22_5b": {
        "model": {"name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "quantize": True, "quantize_te": True},
        "sample": {"width": 480, "height": 480, "num_inference_steps": 20, "guidance_scale": 5.0, "seed": 42, "num_frames": 17},
    },
    "ltx2.3": {
        # even quantized, the 22B stack does not fit a 32GB card — needs the big GPU
        "model": {"name_or_path": "Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 3.0, "seed": 42, "num_frames": 25},
    },
    # single-file / comfy-layout archs: weights resolve under MODELS_PATH (or
    # download there with --allow-download)
    "chroma": {
        "model": {"name_or_path": "lodestones/Chroma1-HD", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 26, "guidance_scale": 4.0},
    },
    "flux_kontext": {
        "model": {"name_or_path": "black-forest-labs/FLUX.1-Kontext-dev", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 20, "guidance_scale": 2.5},
        "needs_control_image": True,
    },
    "flux2_klein_4b": {
        "model": {"name_or_path": "black-forest-labs/FLUX.2-klein-base-4B", "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 25, "guidance_scale": 4.0},
    },
    # ---- coverage for every UI-default arch (big downloads on first run) ----
    "wan22_14b": {
        "model": {"name_or_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16", "quantize": True, "quantize_te": True, "low_vram": True},
        "sample": {"width": 480, "height": 480, "num_inference_steps": 20, "guidance_scale": 3.5, "seed": 42, "num_frames": 17},
    },
    "wan22_14b_i2v": {
        "model": {"name_or_path": "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16", "quantize": True, "quantize_te": True, "low_vram": True},
        "sample": {"width": 480, "height": 480, "num_inference_steps": 20, "guidance_scale": 3.5, "seed": 42, "num_frames": 17},
        "needs_control_image": True,
    },
    "wan21_i2v": {
        "model": {"name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "quantize": True, "quantize_te": True},
        "sample": {"width": 480, "height": 480, "num_inference_steps": 20, "guidance_scale": 5.0, "seed": 42, "num_frames": 17},
        "needs_control_image": True,
    },
    "hidream": {
        "model": {"name_or_path": "HiDream-ai/HiDream-I1-Full", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 28, "guidance_scale": 5.0, "seed": 42},
    },
    "hidream_e1": {
        # editing seq budget fits 768x768 (source+target concat)
        "model": {"name_or_path": "HiDream-ai/HiDream-E1-1", "quantize": True, "quantize_te": True},
        "sample": {"width": 768, "height": 768, "num_inference_steps": 28, "guidance_scale": 5.0, "seed": 42},
        "needs_control_image": True,
        # native editing seq assert: other resolutions hard-fail
        "size_locked": True,
    },
    "nucleus_image": {
        "model": {"name_or_path": "NucleusAI/Nucleus-Image", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "omnigen2": {
        "model": {"name_or_path": "OmniGen2/OmniGen2", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "ltx2.5": {
        "model": {"name_or_path": "Lightricks/LTX-2.5", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 3.0, "seed": 42, "num_frames": 25},
    },
    "flux2": {
        "model": {"name_or_path": "black-forest-labs/FLUX.2-dev", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 25, "guidance_scale": 4.0},
    },
    "flux2_klein_9b": {
        "model": {"name_or_path": "black-forest-labs/FLUX.2-klein-base-9B", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 25, "guidance_scale": 4.0},
    },
    "prx_pixel": {
        "model": {"name_or_path": "Photoroom/prxpixel-t2i", "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 25, "guidance_scale": 4.0},
    },
    "zeta_chroma": {
        "model": {"name_or_path": "lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors", "extras_name_or_path": "Tongyi-MAI/Z-Image-Turbo", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 25, "guidance_scale": 4.0},
    },
    "zimage_l2p": {
        "model": {"name_or_path": "zhen-nan/L2P/model-1k-merge.safetensors", "extras_name_or_path": "Tongyi-MAI/Z-Image-Turbo", "quantize_te": True},
        "sample": {**IMG, "guidance_scale": 1.0},
    },
    "qwen_image_edit": {
        "model": {"name_or_path": "Qwen/Qwen-Image-Edit", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 20, "guidance_scale": 4.0},
        "needs_control_image": True,
    },
    "qwen_image_edit_plus": {
        "model": {"name_or_path": "Qwen/Qwen-Image-Edit-2509", "quantize": True, "quantize_te": True},
        "sample": {**IMG, "num_inference_steps": 20, "guidance_scale": 4.0},
        "needs_control_image": True,
    },
    # ---- legacy monolith archs (components adopted into v2 on load) ----
    "sd1": {
        "model": {"name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5"},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 7.5, "seed": 42},
    },
    "sdxl": {
        "model": {"name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 6.0, "seed": 42},
    },
    "ace_step_15": {
        "model": {"name_or_path": "ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 4.0, "seed": 42},
    },
    "f-lite": {
        "model": {"name_or_path": "Freepik/F-Lite", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
}

SKIP_MARKERS = (
    "couldn't connect",
    "offline mode",
    "hf_hub_offline",
    "cannot find the requested files",
    "not found in cache",
    "localentrynotfounderror",
    "does not appear to have a file named",
    "404 client error",
    "entrynotfounderror",
    "gatedrepoerror",
    "cannot access gated repo",
    "repositorynotfounderror",
)


def classify_error(err: BaseException) -> str:
    text = f"{type(err).__name__}: {err}".lower()
    if any(m in text for m in SKIP_MARKERS):
        return "SKIP"
    if isinstance(err, FileNotFoundError):
        return "SKIP"
    return "FAIL"


def run_one(
    arch: str,
    device: str,
    allow_download: bool,
    qtype_override: str = None,
    quant_only: bool = False,
    no_sample: bool = False,
) -> dict:
    entry = MODEL_TESTS[arch]
    out_dir = os.path.join(OUTPUT_ROOT, arch.replace("/", "_").replace(":", "_"))
    if qtype_override:
        out_dir = os.path.join(out_dir, f"qtype_{qtype_override}")
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*")):
        os.remove(old)

    from toolkit.config_modules import GenerateImageConfig, ModelConfig
    from toolkit.util.get_model import get_model_class

    # quantization always tests the convrot8 backend (per-entry override wins)
    model_kwargs = dict(entry["model"])
    if qtype_override:
        # quant smoke: force quantization of transformer + TE at this qtype
        model_kwargs["quantize"] = True
        model_kwargs["quantize_te"] = True
        model_kwargs["qtype"] = qtype_override
        model_kwargs["qtype_te"] = qtype_override
    else:
        if model_kwargs.get("quantize"):
            model_kwargs.setdefault("qtype", "convrot8")
        if model_kwargs.get("quantize_te"):
            model_kwargs.setdefault("qtype_te", "convrot8")

    model_config = ModelConfig(arch=arch, dtype="bf16", **model_kwargs)
    ModelClass = get_model_class(model_config)
    from toolkit.util.get_model import LEGACY_ARCHS

    if (
        getattr(ModelClass, "arch", None) not in (arch, model_config.arch)
        and model_config.arch not in LEGACY_ARCHS
    ):
        raise ValueError(
            f"arch {arch!r} resolved to {ModelClass.__name__} "
            f"(arch={getattr(ModelClass, 'arch', None)!r}) — registry mismatch"
        )

    sampler = None
    if hasattr(ModelClass, "get_train_scheduler"):
        sampler = ModelClass.get_train_scheduler()
    else:
        # legacy monolith archs build their scheduler the way training does
        from toolkit.sampler import get_sampler

        legacy_arch = "sd"
        if model_config.is_pixart:
            legacy_arch = "pixart"
        elif model_config.is_flux:
            legacy_arch = "flux"
        elif model_config.is_lumina2:
            legacy_arch = "lumina2"
        sampler = get_sampler(
            "ddpm",
            {
                "prediction_type": "v_prediction"
                if model_config.is_v_pred
                else "epsilon",
            },
            arch=legacy_arch,
        )

    sd = ModelClass(
        device=device,
        model_config=model_config,
        dtype="bf16",
        noise_scheduler=sampler,
    )

    # ---- per-stage speed / VRAM / CPU instrumentation + cProfile ----
    import cProfile

    from testing.stage_profiler import StageProfiler, profile_top

    prof = StageProfiler(device)
    if hasattr(sd, "print_and_status_update"):
        # every holder announces its stages through print_and_status_update;
        # each status line opens a new profiler stage
        _orig_status = sd.print_and_status_update

        def _status_hook(msg, *a, **k):
            prof.stage(str(msg))
            return _orig_status(msg, *a, **k)

        sd.print_and_status_update = _status_hook

    prof.stage("load: init")
    load_profile = cProfile.Profile()
    load_profile.enable()
    t_load0 = time.perf_counter()
    sd.load_model()
    load_seconds = time.perf_counter() - t_load0
    load_profile.disable()
    load_profile.dump_stats(os.path.join(out_dir, "load.prof"))

    sample_kwargs = dict(entry["sample"])
    if quant_only:
        # quant smoke: one tiny pass just to prove the quantized forward runs.
        # Kernel-shape bugs live in layer dims (K/N), not token count, so a
        # small image exercises them; 2 steps not 1 (shift_terminal NaNs).
        sample_kwargs["num_inference_steps"] = 2
        if not entry.get("size_locked"):
            # 384 divides every bucket size in the registry (16/32/64)
            sample_kwargs["width"] = min(sample_kwargs["width"], 384)
            sample_kwargs["height"] = min(sample_kwargs["height"], 384)
            if "num_frames" in sample_kwargs:
                sample_kwargs["num_frames"] = min(sample_kwargs["num_frames"], 9)
    if entry.get("needs_control_image"):
        # edit/kontext models require a control image; a flat gray input is fine
        from PIL import Image

        ctrl_path = os.path.join(out_dir, ".ctrl.png")
        Image.new(
            "RGB", (sample_kwargs["width"], sample_kwargs["height"]), (128, 128, 128)
        ).save(ctrl_path)
        sample_kwargs["ctrl_img"] = ctrl_path
    gen = GenerateImageConfig(
        prompt="a photo of a cat sitting on a wooden table",
        output_folder=out_dir,
        # the GenerateImageConfig default for output_ext is the Literal type
        # alias itself; real callers always pass one
        output_ext="png",
        **sample_kwargs,
    )
    gen_kwargs = {}
    if not hasattr(ModelClass, "get_train_scheduler"):
        # the legacy monolith takes the sampler NAME at generate time
        gen_kwargs["sampler"] = "ddpm"

    if quant_only and no_sample:
        # load+quantize only — no forward at all; catches quantize-time
        # errors but NOT broken quantized kernels (those need the sample)
        stages = prof.finish()
        return _finish_result(
            arch, out_dir, model_config, model_kwargs, load_seconds,
            None, None, 0, stages, load_profile, None, None,
            require_output=False,
        )

    prof.stage("generate")
    gen_profile = cProfile.Profile()
    gen_profile.enable()
    t_gen0 = time.perf_counter()
    sd.generate_images([gen], **gen_kwargs)
    gen_seconds = time.perf_counter() - t_gen0
    gen_profile.disable()
    gen_profile.dump_stats(os.path.join(out_dir, "gen.prof"))

    # ---- 100% offload forward: attach full layer offloading to the DiT and
    # text encoder(s), then run a 1-step sample so the offload path (TE
    # encode + denoise + decode with every layer staged from cpu) is proven
    # for every arch ----
    import torch

    offload_seconds = None
    attached = 0
    offload_profile = None
    if quant_only:
        stages = prof.finish()
        return _finish_result(
            arch, out_dir, model_config, model_kwargs, load_seconds,
            gen_seconds, offload_seconds, attached, stages,
            load_profile, gen_profile, offload_profile,
        )

    from toolkit.memory_management import MemoryManager

    prof.stage("offload 100%: attach")
    offload_targets = []
    model_module = getattr(sd, "model", None)
    if model_module is not None:
        subs = [
            getattr(model_module, name, None)
            for name in ("transformer_1", "transformer_2")
        ]
        if all(s is not None for s in subs):
            # dual-DiT wrappers (wan22 14b): pipelines hold and .to() the
            # sub-transformers directly, so each needs its own manager
            offload_targets.extend(subs)
        else:
            offload_targets.append(model_module)
    tes = getattr(sd, "text_encoder", None)
    for te in tes if isinstance(tes, list) else [tes]:
        offload_targets.append(te)
    # auxiliary conditioning stacks where archs have them (ltx connectors,
    # i2v vision towers)
    offload_targets.append(getattr(sd, "image_encoder", None))
    offload_targets.append(getattr(getattr(sd, "pipeline", None), "connectors", None))
    attached = 0
    for m in offload_targets:
        if not isinstance(m, torch.nn.Module):
            continue
        if type(m).__name__.startswith("Fake"):
            continue
        if next(m.parameters(), None) is None:
            continue
        get_ignore = getattr(m, "get_offload_ignore_modules", None)
        ignore = get_ignore() if callable(get_ignore) else None
        MemoryManager.attach(
            m,
            torch.device(device),
            offload_percent=1.0,
            ignore_modules=list(ignore or []),
        )
        attached += 1

    prof.stage("offload 100%: generate")
    offload_profile = cProfile.Profile()
    offload_profile.enable()
    t_off0 = time.perf_counter()
    # 2 steps, not 1: diffusers' shift_terminal stretch divides by
    # one_minus_z[-1], which is zero for a single step — NaN timesteps and a
    # NaN image would mask real numerical breakage in the offload path
    gen_offload = GenerateImageConfig(
        prompt="a photo of a cat sitting on a wooden table",
        output_folder=out_dir,
        output_ext="png",
        **{**sample_kwargs, "num_inference_steps": 2},
    )
    sd.generate_images([gen_offload], **gen_kwargs)
    offload_seconds = time.perf_counter() - t_off0
    offload_profile.disable()
    offload_profile.dump_stats(os.path.join(out_dir, "offload.prof"))

    stages = prof.finish()
    return _finish_result(
        arch, out_dir, model_config, model_kwargs, load_seconds,
        gen_seconds, offload_seconds, attached, stages,
        load_profile, gen_profile, offload_profile,
    )


def _finish_result(
    arch, out_dir, model_config, model_kwargs, load_seconds, gen_seconds,
    offload_seconds, attached, stages, load_profile, gen_profile,
    offload_profile, require_output=True,
):
    import torch

    from testing.stage_profiler import profile_top

    produced = [
        p
        for p in glob.glob(os.path.join(out_dir, "*"))
        if os.path.isfile(p)
        and os.path.getsize(p) > 1024
        and not p.endswith((".txt", ".prof", ".json"))
    ]
    if not produced and require_output:
        raise RuntimeError(f"no output file produced in {out_dir}")

    profile = {
        "load_top": profile_top(load_profile),
        "load_prof": os.path.join(out_dir, "load.prof"),
    }
    if gen_profile is not None:
        profile["gen_top"] = profile_top(gen_profile, limit=15)
        profile["gen_prof"] = os.path.join(out_dir, "gen.prof")
    if offload_profile is not None:
        profile["offload_top"] = profile_top(offload_profile, limit=15)
        profile["offload_prof"] = os.path.join(out_dir, "offload.prof")
    result = {
        "arch": arch,
        "status": "PASS",
        "outputs": produced,
        "qtype": model_config.qtype if model_kwargs.get("quantize") else None,
        "qtype_te": model_config.qtype_te if model_kwargs.get("quantize_te") else None,
        "load_seconds": round(load_seconds, 3),
        "gen_seconds": round(gen_seconds, 3) if gen_seconds is not None else None,
        "offload_seconds": (
            round(offload_seconds, 3) if offload_seconds is not None else None
        ),
        "offload_modules_attached": attached,
        "stages": stages,
        "profile": profile,
        "env": {
            "torch_num_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def format_stage_table(stages) -> str:
    """Aligned markdown table: stage column left-aligned, numbers right-aligned
    with fixed decimals so they line up in the terminal too."""
    headers = [
        "stage", "seconds", "vram peak GB", "vram resv GB", "vram end GB",
        "rss peak GB", "cpu avg %", "cpu max %", "threads",
    ]

    def fmt(value, decimals):
        if value is None or value == "-":
            return "-"
        return f"{value:.{decimals}f}"

    rows = [
        [
            s["name"],
            fmt(s.get("seconds"), 2),
            fmt(s.get("vram_peak_gb", "-"), 2),
            fmt(s.get("vram_reserved_gb", "-"), 2),
            fmt(s.get("vram_end_gb", "-"), 2),
            fmt(s.get("rss_peak_gb"), 2),
            fmt(s.get("cpu_avg_pct"), 1),
            fmt(s.get("cpu_max_pct"), 1),
            str(s.get("threads_max", "-")),
        ]
        for s in stages
    ]

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells):
        out = []
        for i, cell in enumerate(cells):
            # stage name left-aligned, everything else right-aligned
            out.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return "| " + " | ".join(out) + " |"

    sep = "|" + "|".join(
        (":" + "-" * (w + 1)) if i == 0 else ("-" * (w + 1) + ":")
        for i, w in enumerate(widths)
    ) + "|"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def write_report(results, path):
    """Aggregated markdown report: per-arch stage tables + cross-arch summary
    + load-profile hotspots, for diagnosing load/quantization speedups and
    CPU usage."""

    def _sum(stages, pred):
        return round(sum(s["seconds"] for s in stages if pred(s)), 1)

    lines = ["# Model loading test report", ""]
    lines.append(
        "Per-arch stage timings, peak VRAM (torch allocated/reserved), peak "
        "process RSS, and CPU utilization (100 = one core, sampled at 50ms). "
        "Quantization is convrot8 across the board. Full cProfile dumps sit "
        "next to each arch's outputs (load.prof / gen.prof; inspect with "
        "`python -m pstats <file>` or snakeviz)."
    )
    lines.append("")

    # ---- summary ----
    lines += ["## Summary", ""]
    lines += [
        "| arch | status | load s | quantize s | generate s | vram peak GB | rss peak GB | quant cpu avg % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] != "PASS" or "stages" not in r:
            lines.append(
                f"| {r['arch']} | {r['status']} | - | - | - | - | - | - | "
            )
            continue
        stages = r["stages"]
        quant = [s for s in stages if "uantiz" in s["name"]]
        vram_peak = max((s.get("vram_peak_gb", 0) for s in stages), default=0)
        rss_peak = max((s.get("rss_peak_gb", 0) for s in stages), default=0)
        quant_cpu = (
            round(sum(s["cpu_avg_pct"] for s in quant) / len(quant), 1)
            if quant
            else "-"
        )
        lines.append(
            f"| {r['arch']} | PASS | {r['load_seconds']} "
            f"| {_sum(quant, lambda s: True)} | {r['gen_seconds']} "
            f"| {vram_peak} | {rss_peak} | {quant_cpu} |"
        )
    lines.append("")

    # ---- per-arch detail ----
    for r in results:
        if r["status"] != "PASS" or "stages" not in r:
            lines += [f"## {r['arch']} — {r['status']}", "", r.get("error", ""), ""]
            continue
        lines += [f"## {r['arch']}", ""]
        lines += [
            f"load {r['load_seconds']}s, generate {r['gen_seconds']}s, "
            f"qtype {r.get('qtype')}, qtype_te {r.get('qtype_te')}, "
            f"torch threads {r['env']['torch_num_threads']}/{r['env']['cpu_count']} cores",
            "",
        ]
        lines += [format_stage_table(r["stages"]), ""]
        top = r.get("profile", {}).get("load_top", [])[:12]
        if top:
            lines += ["Load hotspots (cumulative):", "", "```"]
            for row in top:
                lines.append(
                    f"{row['cumtime']:>9.2f}s  tot {row['tottime']:>8.2f}s  "
                    f"x{row['ncalls']:<8} {row['func']}"
                )
            lines += ["```", ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--json-result", type=str, default=None)
    parser.add_argument(
        "--quant-test",
        action="store_true",
        help="quant smoke: for each arch (or --arch), force-quantize the "
        "transformer + TE at each qtype in --qtypes, run one tiny 2-step "
        "pass, no offload probe — just proves each backend loads/quantizes/"
        "runs without errors",
    )
    parser.add_argument(
        "--qtypes",
        type=str,
        default="convrot8,qfloat8,float8",
        help="comma-separated qtypes for --quant-test",
    )
    parser.add_argument("--qtype-override", type=str, default=None)
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="with --quant-test: load+quantize only, skip the 2-step sample "
        "(faster, but does not exercise the quantized kernels)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild report.md from each arch's saved metrics.json, no runs",
    )
    args = parser.parse_args()

    if args.list:
        for arch in MODEL_TESTS:
            print(arch)
        return

    if args.report_only:
        results = []
        for arch in MODEL_TESTS:
            out_dir = os.path.join(
                OUTPUT_ROOT, arch.replace("/", "_").replace(":", "_")
            )
            metrics_path = os.path.join(out_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    results.append(json.load(f))
            else:
                results.append(
                    {"arch": arch, "status": "SKIP", "error": "no metrics.json saved"}
                )
        report_path = write_report(results, os.path.join(OUTPUT_ROOT, "report.md"))
        print(f"report: {report_path}")
        return

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.quant_test and args.qtype_override is None:
        # quant smoke: (archs x qtypes) grid, one subprocess per cell so each
        # backend gets a clean CUDA context
        qtypes = [q.strip() for q in args.qtypes.split(",") if q.strip()]
        archs = [args.arch] if args.arch else list(MODEL_TESTS)
        results = []
        total = len(archs) * len(qtypes)
        i = 0
        t0 = time.perf_counter()
        for arch in archs:
            for qt in qtypes:
                i += 1
                print(f"\n===== {arch} [{qt}] ({i}/{total}) =====", flush=True)
                result_path = os.path.join(
                    OUTPUT_ROOT, f".{arch.replace('/', '_')}.{qt}.result.json"
                )
                cmd = [
                    sys.executable, os.path.abspath(__file__),
                    "--arch", arch, "--device", args.device,
                    "--qtype-override", qt, "--quant-test",
                    "--json-result", result_path,
                ]
                if args.allow_download:
                    cmd.append("--allow-download")
                if args.no_sample:
                    cmd.append("--no-sample")
                subprocess.run(cmd, cwd=TOOLKIT_ROOT)
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        r = json.load(f)
                    os.remove(result_path)
                else:
                    r = {"arch": arch, "status": "FAIL", "error": "subprocess died"}
                r["qtype_tested"] = qt
                results.append(r)
                elapsed = time.perf_counter() - t0
                print(
                    f">>> quant-test progress: {i}/{total} — elapsed "
                    f"{elapsed / 60:.0f}m, eta ~{(elapsed / i) * (total - i) / 60:.0f}m",
                    flush=True,
                )
        print("\n===== quant-test summary =====")
        fails = 0
        for r in results:
            line = f"[{r['status']}] {r['arch']} [{r['qtype_tested']}]"
            if r["status"] != "PASS":
                line += f" — {r.get('error', '')[:140]}"
                fails += r["status"] == "FAIL"
            print(line)
        if fails:
            sys.exit(1)
        return

    if args.arch is not None:
        if args.arch not in MODEL_TESTS:
            raise SystemExit(
                f"arch {args.arch!r} is not registered; --list shows options"
            )
        try:
            result = run_one(
                args.arch,
                args.device,
                args.allow_download,
                qtype_override=args.qtype_override,
                quant_only=args.quant_test,
                no_sample=args.no_sample,
            )
        except BaseException as err:
            status = classify_error(err)
            result = {"arch": args.arch, "status": status, "error": f"{type(err).__name__}: {err}"}
            if status == "FAIL":
                import traceback

                traceback.print_exc()
        if args.json_result:
            with open(args.json_result, "w") as f:
                json.dump(result, f)
        if result["status"] == "PASS" and "stages" in result:
            print()
            print(format_stage_table(result["stages"]))
            print()
        print(f"[{result['status']}] {args.arch}" + (f" — {result.get('error', '')}" if result["status"] != "PASS" else ""))
        if result["status"] == "FAIL":
            sys.exit(1)
        return

    if not args.all:
        parser.print_help()
        return

    # --all: one subprocess per arch so every model fully unloads (clean CUDA
    # teardown) before the next loads
    results = []
    total = len(MODEL_TESTS)
    sweep_t0 = time.perf_counter()
    for i, arch in enumerate(MODEL_TESTS, start=1):
        # flush so tail -f shows headers/progress live when stdout is a file
        print(f"\n===== {arch} ({i}/{total}) =====", flush=True)
        result_path = os.path.join(OUTPUT_ROOT, f".{arch.replace('/', '_')}.result.json")
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--arch",
            arch,
            "--device",
            args.device,
            "--json-result",
            result_path,
        ]
        if args.allow_download:
            cmd.append("--allow-download")
        proc = subprocess.run(cmd, cwd=TOOLKIT_ROOT)
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))
            os.remove(result_path)
        else:
            results.append(
                {"arch": arch, "status": "FAIL", "error": f"subprocess died (exit {proc.returncode})"}
            )

        # running tally so a long sweep always shows how far along it is
        tally = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for r in results:
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        elapsed = time.perf_counter() - sweep_t0
        eta = (elapsed / i) * (total - i)
        print(
            f">>> progress: {i}/{total} ({i * 100 // total}%) — "
            f"{tally['PASS']} pass, {tally['FAIL']} fail, {tally['SKIP']} skip — "
            f"elapsed {elapsed / 60:.0f}m, eta ~{eta / 60:.0f}m",
            flush=True,
        )

    print("\n===== summary =====")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        line = f"[{r['status']}] {r['arch']}"
        if r["status"] != "PASS":
            line += f" — {r.get('error', '')[:160]}"
        print(line)
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")

    report_path = write_report(results, os.path.join(OUTPUT_ROOT, "report.md"))
    print(f"report: {report_path}")
    if counts["FAIL"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
