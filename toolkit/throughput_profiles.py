import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

try:
    import torch
except Exception:  # pragma: no cover - allows tests without torch installed
    torch = None

if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig, ModelConfig, TrainConfig


@dataclass
class GPUCapability:
    available: bool
    index: int = 0
    name: str = ""
    total_vram_gb: float = 0.0
    sm_count: int = 0
    compute_capability: tuple[int, int] = (0, 0)
    memory_bus_width_bits: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    bandwidth_hint_gbps: Optional[float] = None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_device_index(device: Optional[object]) -> int:
    if device is None:
        return 0
    if isinstance(device, int):
        return max(0, int(device))
    if isinstance(device, str):
        if device.startswith("cuda:"):
            try:
                return max(0, int(device.split(":", 1)[1]))
            except Exception:
                return 0
        if device.isdigit():
            return max(0, int(device))
        return 0

    # Avoid hard torch type dependency in fallback-only environments.
    try:
        dev_type = getattr(device, "type", None)
        dev_index = getattr(device, "index", None)
        if dev_type == "cuda" and dev_index is not None:
            return max(0, int(dev_index))
    except Exception:
        pass
    return 0


def _parse_cc(raw: Optional[str]) -> tuple[int, int]:
    if raw is None:
        return (0, 0)
    text = str(raw).strip().lower()
    if text in {"", "n/a", "[not supported]"}:
        return (0, 0)
    try:
        if "." in text:
            major_s, minor_s = text.split(".", 1)
            return (int(major_s), int(minor_s))
        return (int(text), 0)
    except Exception:
        return (0, 0)


def _detect_gpu_capability_via_nvidia_smi(device_index: int) -> Optional[GPUCapability]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.bus_width,memory.clock,compute_cap",
            "--format=csv,noheader,nounits",
            "-i",
            str(max(0, int(device_index))),
        ]
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.5,
        ).strip()
        if output == "":
            return None

        parts = [x.strip() for x in output.split(",")]
        if len(parts) < 2:
            return None

        name = parts[0]
        total_mib = float(parts[1])
        bus_width = None
        mem_clock = None
        cc = (0, 0)

        if len(parts) >= 3:
            try:
                bus_width = int(parts[2])
            except Exception:
                bus_width = None
        if len(parts) >= 4:
            try:
                mem_clock = int(parts[3])
            except Exception:
                mem_clock = None
        if len(parts) >= 5:
            cc = _parse_cc(parts[4])

        capability = GPUCapability(
            available=True,
            index=max(0, int(device_index)),
            name=name,
            total_vram_gb=float(total_mib) / 1024.0,
            sm_count=0,
            compute_capability=cc,
            memory_bus_width_bits=bus_width,
            memory_clock_mhz=mem_clock,
            bandwidth_hint_gbps=None,
        )
        if bus_width is not None and mem_clock is not None:
            capability.bandwidth_hint_gbps = (2.0 * mem_clock * (bus_width / 8.0)) / 1000.0
        return capability
    except Exception:
        return None


def detect_gpu_capability(device: Optional[object] = None) -> GPUCapability:
    device_index = _resolve_device_index(device)
    if torch is None or not torch.cuda.is_available():
        fallback_capability = _detect_gpu_capability_via_nvidia_smi(device_index)
        if fallback_capability is not None:
            return fallback_capability
        return GPUCapability(available=False, index=device_index)

    try:
        device_index = max(0, min(device_index, torch.cuda.device_count() - 1))
        props = torch.cuda.get_device_properties(device_index)
        capability = GPUCapability(
            available=True,
            index=device_index,
            name=str(props.name),
            total_vram_gb=float(props.total_memory) / (1024.0 ** 3),
            sm_count=int(props.multi_processor_count),
            compute_capability=(int(props.major), int(props.minor)),
        )

        # Optional nvidia-smi hinting: bus width + memory clock -> rough bandwidth tiering.
        smi_capability = _detect_gpu_capability_via_nvidia_smi(device_index)
        if smi_capability is not None:
            capability.memory_bus_width_bits = smi_capability.memory_bus_width_bits
            capability.memory_clock_mhz = smi_capability.memory_clock_mhz
            capability.bandwidth_hint_gbps = smi_capability.bandwidth_hint_gbps
            if capability.compute_capability == (0, 0):
                capability.compute_capability = smi_capability.compute_capability
        return capability
    except Exception:
        fallback_capability = _detect_gpu_capability_via_nvidia_smi(device_index)
        if fallback_capability is not None:
            return fallback_capability
        return GPUCapability(available=False, index=device_index)


def resolve_ltx23_throughput_profile(
    requested_profile: str,
    capability: GPUCapability,
) -> str:
    allowed = {"auto", "ltx23_safe", "ltx23_max", "ltx23_ultra_vram"}
    req = (requested_profile or "auto").strip().lower()
    if req not in allowed:
        req = "auto"

    if req != "auto":
        return req

    if not capability.available:
        return "ltx23_safe"

    lower_name = capability.name.lower()
    if "rtx pro 6000" in lower_name or capability.total_vram_gb >= 44.0:
        return "ltx23_ultra_vram"

    # High-end tier, includes 5090-class hardware.
    is_high_end = (
        capability.total_vram_gb >= 22.0
        and (
            capability.sm_count >= 100
            or capability.compute_capability[0] >= 9
        )
    )
    if is_high_end:
        return "ltx23_max"

    return "ltx23_safe"


def get_ltx23_profile_settings(profile: str, cpu_count: Optional[int] = None) -> dict:
    cpu = max(2, int(cpu_count or os.cpu_count() or 8))
    half_cpu = max(2, cpu // 2)
    if profile == "ltx23_ultra_vram":
        workers = _clamp(half_cpu, 10, 20)
        return {
            "num_workers": workers,
            "prefetch_factor": 4,
            "prefetch_queue_depth": 3,
            "logger_commit_interval": 12,
            "compile_mode": "max-autotune",
            "compile_dynamic": True,
            "compile_fullgraph": True,
            "prefetch_to_device": True,
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "force_low_vram_off": True,
        }
    if profile == "ltx23_max":
        workers = _clamp(half_cpu, 8, 16)
        return {
            "num_workers": workers,
            "prefetch_factor": 4 if workers <= 12 else 3,
            "prefetch_queue_depth": 2,
            "logger_commit_interval": 10,
            "compile_mode": "max-autotune",
            "compile_dynamic": True,
            "compile_fullgraph": True,
            "prefetch_to_device": True,
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "force_low_vram_off": False,
        }

    workers = _clamp(half_cpu, 4, 8)
    return {
        "num_workers": workers,
        "prefetch_factor": 2 if workers <= 6 else 3,
        "prefetch_queue_depth": 1,
        "logger_commit_interval": 5,
        "compile_mode": "reduce-overhead",
        "compile_dynamic": True,
        "compile_fullgraph": False,
        "prefetch_to_device": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "force_low_vram_off": False,
    }


def apply_ltx23_throughput_profile(
    train_config: "TrainConfig",
    model_config: "ModelConfig",
    dataset_configs: List["DatasetConfig"],
    device: Optional[object] = None,
) -> tuple[str, GPUCapability]:
    capability = detect_gpu_capability(device=device)
    resolved_profile = resolve_ltx23_throughput_profile(
        requested_profile=getattr(train_config, "throughput_profile", "auto"),
        capability=capability,
    )
    settings = get_ltx23_profile_settings(resolved_profile)

    # Persist resolved values for logging/bench tooling.
    train_config.resolved_throughput_profile = resolved_profile
    train_config.detected_gpu_name = capability.name
    train_config.detected_gpu_vram_gb = round(capability.total_vram_gb, 2)
    train_config.detected_gpu_sm_count = capability.sm_count
    train_config.detected_gpu_compute_capability = capability.compute_capability

    if getattr(train_config, "dataloader_autotune", True):
        for dataset in dataset_configs:
            dataset_workers = int(getattr(dataset, "num_workers", 0) or 0)
            dataset_prefetch = int(getattr(dataset, "prefetch_factor", 0) or 0)
            if resolved_profile == "ltx23_ultra_vram":
                # no-penalty: ultra-vram profile should not reduce existing user concurrency values
                dataset.num_workers = max(dataset_workers, settings["num_workers"])
                dataset.prefetch_factor = max(dataset_prefetch, settings["prefetch_factor"])
            else:
                dataset.num_workers = settings["num_workers"]
                dataset.prefetch_factor = settings["prefetch_factor"]
            dataset.pin_memory = True
            dataset.persistent_workers = True
        train_config.dataloader_pin_memory = True
        train_config.dataloader_persistent_workers = True

    if not getattr(train_config, "_prefetch_to_device_requested", False):
        train_config.prefetch_to_device = settings["prefetch_to_device"]
    if not getattr(train_config, "_prefetch_queue_depth_requested", False):
        if resolved_profile == "ltx23_ultra_vram":
            train_config.prefetch_queue_depth = max(
                int(getattr(train_config, "prefetch_queue_depth", 1) or 1),
                settings["prefetch_queue_depth"],
            )
        else:
            train_config.prefetch_queue_depth = settings["prefetch_queue_depth"]
    if not getattr(train_config, "_logger_commit_interval_requested", False):
        if resolved_profile == "ltx23_ultra_vram":
            train_config.logger_commit_interval = max(
                int(getattr(train_config, "logger_commit_interval", 1) or 1),
                settings["logger_commit_interval"],
            )
        else:
            train_config.logger_commit_interval = settings["logger_commit_interval"]
    if not getattr(train_config, "_allow_tf32_requested", False):
        train_config.allow_tf32 = settings["allow_tf32"]
    if not getattr(train_config, "_cudnn_benchmark_requested", False):
        train_config.cudnn_benchmark = settings["cudnn_benchmark"]

    if resolved_profile in {"ltx23_max", "ltx23_ultra_vram"} and not getattr(model_config, "compile_requested", False):
        model_config.compile = True
    if not getattr(model_config, "compile_mode_requested", False):
        model_config.compile_mode = settings["compile_mode"]
    if not getattr(model_config, "compile_dynamic_requested", False):
        model_config.compile_dynamic = settings["compile_dynamic"]
    if not getattr(model_config, "compile_fullgraph_requested", False):
        model_config.compile_fullgraph = settings["compile_fullgraph"]

    if settings.get("force_low_vram_off", False) and not getattr(model_config, "low_vram_requested", False):
        model_config.low_vram = False

    return resolved_profile, capability


def is_ltx_only_mode_enabled() -> bool:
    if _parse_env_bool("AITK_ALLOW_NON_LTX", False):
        return False
    return _parse_env_bool("AITK_LTX_ONLY_MODE", True)
