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
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "ernie_image": {
        "model": {"name_or_path": "baidu/ERNIE-Image", "quantize": True, "quantize_te": True},
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "mageflow": {
        "model": {"name_or_path": "microsoft/Mage-Flow-Base", "quantize": True, "quantize_te": True},
        "sample": IMG,
    },
    "ideogram4": {
        "model": {"name_or_path": "ideogram-ai/ideogram-4-fp8", "quantize": True, "quantize_te": True},
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "hidream_o1": {
        "model": {"name_or_path": "HiDream-ai/HiDream-O1-Image", "quantize": True, "quantize_te": True},
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 28, "guidance_scale": 5.0, "seed": 42},
    },
    "anima": {
        "model": {"name_or_path": "circlestone-labs/Anima-Base-v1.0-Diffusers"},
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.5, "seed": 42},
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
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 28, "guidance_scale": 5.0, "seed": 42},
    },
    "hidream_e1": {
        # editing seq budget fits 768x768 (source+target concat)
        "model": {"name_or_path": "HiDream-ai/HiDream-E1-1", "quantize": True, "quantize_te": True},
        "sample": {"width": 768, "height": 768, "num_inference_steps": 28, "guidance_scale": 5.0, "seed": 42},
        "needs_control_image": True,
    },
    "nucleus_image": {
        "model": {"name_or_path": "NucleusAI/Nucleus-Image", "quantize": True, "quantize_te": True},
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
    },
    "omnigen2": {
        "model": {"name_or_path": "OmniGen2/OmniGen2", "quantize": True, "quantize_te": True},
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
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
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 6.0, "seed": 42},
    },
    "ace_step_15": {
        "model": {"name_or_path": "ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors", "quantize": True, "quantize_te": True},
        "sample": {"width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 4.0, "seed": 42},
    },
    "f-lite": {
        "model": {"name_or_path": "Freepik/F-Lite", "quantize": True, "quantize_te": True},
        "sample": {"width": 1024, "height": 1024, "num_inference_steps": 25, "guidance_scale": 4.0, "seed": 42},
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


def run_one(arch: str, device: str, allow_download: bool) -> dict:
    entry = MODEL_TESTS[arch]
    out_dir = os.path.join(OUTPUT_ROOT, arch.replace("/", "_").replace(":", "_"))
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*")):
        os.remove(old)

    from toolkit.config_modules import GenerateImageConfig, ModelConfig
    from toolkit.util.get_model import get_model_class

    model_config = ModelConfig(arch=arch, dtype="bf16", **entry["model"])
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
    sd.load_model()

    sample_kwargs = dict(entry["sample"])
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
    sd.generate_images([gen], **gen_kwargs)

    produced = [
        p
        for p in glob.glob(os.path.join(out_dir, "*"))
        if os.path.isfile(p) and os.path.getsize(p) > 1024 and not p.endswith(".txt")
    ]
    if not produced:
        raise RuntimeError(f"no output file produced in {out_dir}")
    return {"arch": arch, "status": "PASS", "outputs": produced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--json-result", type=str, default=None)
    args = parser.parse_args()

    if args.list:
        for arch in MODEL_TESTS:
            print(arch)
        return

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.arch is not None:
        if args.arch not in MODEL_TESTS:
            raise SystemExit(
                f"arch {args.arch!r} is not registered; --list shows options"
            )
        try:
            result = run_one(args.arch, args.device, args.allow_download)
        except BaseException as err:
            status = classify_error(err)
            result = {"arch": args.arch, "status": status, "error": f"{type(err).__name__}: {err}"}
            if status == "FAIL":
                import traceback

                traceback.print_exc()
        if args.json_result:
            with open(args.json_result, "w") as f:
                json.dump(result, f)
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
    for arch in MODEL_TESTS:
        print(f"\n===== {arch} =====")
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

    print("\n===== summary =====")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        line = f"[{r['status']}] {r['arch']}"
        if r["status"] != "PASS":
            line += f" — {r.get('error', '')[:160]}"
        print(line)
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    if counts["FAIL"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
