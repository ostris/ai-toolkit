"""Opt-in real-model validation, using an independent dataset/output directory.

Example (offline, in the toolkit environment):
  python testing/validate_qwen21_training.py --config /source-config.yaml \
      --dataset /validation/dataset --output /validation/results --steps 3

The dataset MUST be a disposable copy: the normal toolkit writes caches there.
Never point this command at a production dataset. The input config is only read;
all generated configuration, checkpoints and metrics go to --output. No UI job
ID or original SQLite path is used. Model weights must already be available.
"""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "YES"
os.environ.pop("AITK_JOB_ID", None)

import torch
import yaml

from testing.test_qwen21_prompt_encoder import legacy_encoder
from toolkit.job import get_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--baseline-batch", type=int, default=1,
                        help="1-based cache batch to compare with the original LM call")
    parser.add_argument("--processor", type=Path, help="Optional local copy of the same processor for offline loading")
    args = parser.parse_args()
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    original = copy.deepcopy(config)
    process = config["config"]["process"][0]
    if process["model"]["arch"] != "qwen_image_2" or len(process["datasets"]) != 1:
        raise ValueError("Expected one qwen_image_2 training dataset")
    if args.dataset.resolve() == Path(process["datasets"][0]["folder_path"]).resolve():
        raise ValueError("--dataset must be a separate disposable dataset copy")
    if (args.dataset / "_t_e_cache").exists():
        raise ValueError("Use a fresh dataset copy without an embedding cache")
    if args.steps < 3:
        raise ValueError("At least three optimizer steps are required")
    if args.baseline_batch < 1:
        raise ValueError("--baseline-batch must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    config["config"]["name"] = "qwen21-validation"
    process["training_folder"] = str(args.output)
    process["sqlite_db_path"] = str(args.output / "unused-sqlite-path.db")
    process["datasets"][0]["folder_path"] = str(args.dataset)
    process["train"]["steps"] = args.steps
    process["save"].update(save_every=1, max_step_saves_to_keep=args.steps + 1, push_to_hub=False)
    process["logging"]["use_ui_logger"] = False
    # Keep rank, resolution, captions, quantization, offloading and all other
    # training settings unchanged. Record both configs locally for review.
    (args.output / "source-config.yaml").write_text(yaml.safe_dump(original))
    (args.output / "validation-config.yaml").write_text(yaml.safe_dump(config))
    metrics = dict(phase="loading", embedding_calls=[], optimizer_steps=[], passed=False)
    stop = threading.Event()

    def persist():
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def monitor():
        while not stop.wait(0.1):
            free, total = torch.cuda.mem_get_info()
            key = metrics["phase"] + "_sampled_device_used_peak_bytes"
            metrics[key] = max(metrics.get(key, 0), total - free)

    torch.cuda.init()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import QwenImage21PromptEncoder
    from extensions_built_in.diffusion_models.qwen_image_2.src.text_encoder import QwenImage21TextEncoder
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
    encode = QwenImage21PromptEncoder.encode
    before_loop = DiffusionTrainer.hook_before_train_loop
    compared = False
    batch_number = 0

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), device_used_bytes=total - free)

    def flush():
        gc.collect()
        torch.cuda.empty_cache()

    def measured_encode(self, prompts, images=None, device=None):
        nonlocal compared, batch_number
        baseline_reference = None
        if len(prompts) > 1:
            batch_number += 1
        if len(prompts) > 1 and not compared and batch_number == args.baseline_batch:
            compared = True
            metrics["phase"] = "baseline"
            torch.cuda.reset_peak_memory_stats()
            try:
                reference = legacy_encoder(self.text_encoder, self.processor).encode(prompts, images, device)
                metrics["baseline_batch"] = dict(batch=len(prompts), cache_batch_index=batch_number, oom=False, **memory())
                baseline_reference = [[tensor.cpu().clone() for tensor in group] for group in reference]
                del reference
            except torch.OutOfMemoryError as error:
                metrics["baseline_batch"] = dict(batch=len(prompts), cache_batch_index=batch_number, oom=True, error=str(error), **memory())
            flush()
            # Compare the same single prompt before/after. Comparing differently
            # padded batches would test different kernels and positions instead.
            ref = legacy_encoder(self.text_encoder, self.processor).encode(
                prompts[-1:], images[-1:] if images is not None else None, device
            )
            ref = [[tensor.cpu().clone() for tensor in group] for group in ref]
            flush()
            actual = encode(self, prompts[-1:], images[-1:] if images is not None else None, device)
            max_difference = 0.0
            for expected_group, actual_group in zip(ref, actual):
                for expected, value in zip(expected_group, actual_group):
                    value = value.cpu()
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    if value.is_floating_point():
                        max_difference = max(max_difference, (value.float() - expected.float()).abs().max().item())
            metrics["real_model_embedding_comparison"] = dict(max_abs_difference=max_difference, rtol=0, atol=0)
            del ref, actual
            flush()
        metrics["phase"] = "embeddings"
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = encode(self, prompts, images, device)
        if baseline_reference is not None:
            for expected_group, actual_group in zip(baseline_reference, result):
                for expected, value in zip(expected_group, actual_group):
                    torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
            metrics["real_model_full_batch_exact_match"] = True
        metrics["embedding_calls"].append(dict(
            batch=len(prompts), lengths=[len(t) for t in result[0]],
            seconds=time.monotonic() - started, **memory(),
        ))
        persist()
        return result

    def instrument_training(self):
        before_loop(self)
        # Register on the underlying optimizer after Accelerate's wrapping.
        optimizer = self.optimizer
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        trainable = {name: param for name, param in self.network.named_parameters() if param.requires_grad}
        initial = {name: param.detach().cpu().clone() for name, param in trainable.items()}
        metrics["lora_trainable_tensors"] = len(initial)
        metrics["phase"] = "training"
        torch.cuda.reset_peak_memory_stats()

        def after_step(optimizer, args, kwargs):
            changed, maximum = 0, 0.0
            for name, param in trainable.items():
                current = param.detach().cpu()
                if not bool(torch.isfinite(current).all()):
                    raise AssertionError("Non-finite LoRA weights")
                changed += int(not torch.equal(current, initial[name]))
                maximum = max(maximum, (current.float() - initial[name].float()).abs().max().item())
            metrics["optimizer_steps"].append(dict(
                step=len(metrics["optimizer_steps"]) + 1,
                changed_tensors_from_initial=changed, max_abs_delta_from_initial=maximum,
                **memory(),
            ))
            persist()
            print("QWEN21_VALIDATION_STEP", json.dumps(metrics["optimizer_steps"][-1]), flush=True)

        optimizer.register_step_post_hook(after_step)

    QwenImage21PromptEncoder.encode = measured_encode
    DiffusionTrainer.hook_before_train_loop = instrument_training
    processor_patch = None
    try:
        if args.processor is not None:
            # Keep the model/weight resolver unchanged; only source the same
            # processor metadata from a complete local directory when needed.
            processor = QwenImage21TextEncoder.load_processor(str(args.processor), subfolder="")
            processor_patch = mock.patch.object(QwenImage21TextEncoder, "load_processor", return_value=processor)
            processor_patch.start()
            metrics["processor_override"] = str(args.processor)
        job = get_job(config)
        job.run()
        cache_files = list((args.dataset / "_t_e_cache").glob("*.safetensors"))
        metrics["embedding_cache_files"] = len(cache_files)
        metrics["image_count"] = sum(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in args.dataset.iterdir())
        if len(cache_files) != metrics["image_count"]:
            raise AssertionError("Missing cached embeddings")
        if len(metrics["optimizer_steps"]) < args.steps:
            raise AssertionError("Not all optimizer steps completed (possibly skipped OOM batches)")
        if not all(step["changed_tensors_from_initial"] > 0 for step in metrics["optimizer_steps"]):
            raise AssertionError("LoRA weights did not change")
        if not compared:
            raise AssertionError("Requested baseline batch was not reached")
        metrics["passed"] = True
        job.cleanup()
    except BaseException as error:
        metrics["error"] = repr(error)
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        persist()
        QwenImage21PromptEncoder.encode = encode
        DiffusionTrainer.hook_before_train_loop = before_loop
        if processor_patch is not None:
            processor_patch.stop()
        print("QWEN21_VALIDATION_RESULT", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
