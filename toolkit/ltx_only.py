from toolkit.throughput_profiles import is_ltx_only_mode_enabled


def is_ltx_training_process(process_cfg: dict) -> bool:
    if not isinstance(process_cfg, dict):
        return True
    model_cfg = process_cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        return True
    arch = str(model_cfg.get("arch", "") or "").strip().lower()
    name_or_path = str(model_cfg.get("name_or_path", "") or "").strip().lower()
    if arch.startswith("ltx2"):
        return True
    if "lightricks/ltx-2.3" in name_or_path or "ltx-2.3" in name_or_path:
        return True
    return False


def validate_ltx_only_config(config: dict):
    if not is_ltx_only_mode_enabled():
        return
    process_list = config.get("config", {}).get("process", [])
    for idx, process_cfg in enumerate(process_list):
        process_type = str(process_cfg.get("type", "")).strip().lower()
        if process_type not in {"sd_trainer", "diffusion_trainer", "ui_trainer"}:
            continue
        if not is_ltx_training_process(process_cfg):
            raise ValueError(
                "LTX-only mode is enabled. Non-LTX training jobs are blocked. "
                f"Found non-LTX process at config.process[{idx}] (type={process_type}). "
                "Set AITK_ALLOW_NON_LTX=1 to override."
            )
