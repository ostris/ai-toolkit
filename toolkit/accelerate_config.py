from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_default_accelerate_config(
    num_gpus: int = 2, mixed_precision: str = "fp16"
) -> Dict[str, Any]:
    """Generate default Accelerate configuration for multi-GPU training."""
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "downcast_bf16": "no",
        "gpu_ids": ",".join(str(i) for i in range(num_gpus)),
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": mixed_precision,
        "num_machines": 1,
        "num_processes": num_gpus,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }


def save_accelerate_config(
    config: Dict[str, Any], config_path: Optional[str] = None
) -> str:
    """Save Accelerate configuration to file."""
    if config_path is None:
        config_path = Path.cwd() / "accelerate_config.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(config_path)


def load_accelerate_config(config_path: str) -> Dict[str, Any]:
    """Load Accelerate configuration from file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_accelerate_config(config: Dict[str, Any]) -> bool:
    """Validate Accelerate configuration."""
    required_fields = [
        "compute_environment",
        "distributed_type",
        "mixed_precision",
        "num_processes",
        "gpu_ids",
    ]

    for field in required_fields:
        if field not in config:
            return False

    return True


def create_multi_gpu_config(num_gpus: int, mixed_precision: str = "fp16") -> str:
    """Create and save a multi-GPU Accelerate configuration."""
    config = get_default_accelerate_config(num_gpus, mixed_precision)
    config_path = save_accelerate_config(config)
    return config_path
