"""ComfyUI-layout weight file resolution.

Weight files live under MODELS_PATH in ComfyUI's folder layout
(diffusion_models/, text_encoders/, vae/, ...) so the folder is shareable with
a ComfyUI install. Files are used in place when present and downloaded to
exactly their repo-relative location only when missing, so nothing is ever
duplicated on re-run.

Lifted from the minimax_h3 / ltx2.5 model implementations; those now call
into here.
"""

import os
from typing import Callable, Iterable, Optional

from toolkit.paths import MODELS_PATH


def comfy_precision_rank(filename: str, qtype: Optional[str] = None) -> int:
    """Load-preference rank for a comfy weight filename, given the REQUESTED
    quantization. A file whose shipped quantization matches the request loads
    with no work; anything else costs a dequantize/requantize pass, so:

    - convrot* requested: convrot (0) > fp8 mixed (1) > fp8 (2) > bf16 (3) >
      fp16 (4) > other (5)
    - float8/qfloat8 requested: fp8 mixed (0) > fp8 (1) > bf16 (2) > fp16 (3)
      > convrot (4) > other (5)
    - nvfp4 requested: nvfp4 (0) > bf16 (1) > fp16 (2) > convrot (3) > fp8
      (4) > other (5)
    - no quantization requested (full precision) or any other fresh-quant
      backend: bf16 (0) > fp16 (1) > convrot (2) > fp8 mixed (3) > fp8 (4) >
      other (5) — clean weights beat paying a dequantize
    """
    name = os.path.basename(filename).lower()
    is_convrot = "convrot" in name
    is_nvfp4 = "nvfp4" in name
    is_fp8 = ("fp8" in name or "float8" in name or "e4m3" in name) and not is_nvfp4
    is_fp8_mixed = is_fp8 and "mixed" in name
    is_bf16 = "bf16" in name
    is_fp16 = "fp16" in name and not is_fp8

    qt = (qtype or "").lower()
    if qt.startswith("convrot"):
        order = [is_convrot, is_fp8_mixed, is_fp8, is_bf16, is_fp16]
    elif "float8" in qt or qt == "qfloat8":
        order = [is_fp8_mixed, is_fp8, is_bf16, is_fp16, is_convrot]
    elif "nvfp4" in qt:
        order = [is_nvfp4, is_bf16, is_fp16, is_convrot, is_fp8]
    else:
        order = [is_bf16, is_fp16, is_convrot, is_fp8_mixed, is_fp8]
    for rank, flag in enumerate(order):
        if flag:
            return rank
    return 5


def comfy_local_rel(repo_rel: str) -> str:
    """Repo file path -> ComfyUI models-folder path. Comfy-Org repos nest the
    comfy layout under a packaging prefix (split_files/, non_official/) that is
    not part of the shared models folder layout."""
    for prefix in ("split_files/", "non_official/"):
        if repo_rel.startswith(prefix):
            return repo_rel[len(prefix):]
    return repo_rel


def resolve_comfy_candidates(
    candidates: Iterable[str],
    repo_id: str,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
    local_only: bool = False,
    qtype: Optional[str] = None,
) -> Optional[str]:
    """Pick the best comfy weight file among precision variants of one
    component (repo-relative paths, ranked by comfy_precision_rank for the
    requested qtype, then list order). The best-ranked LOCAL candidate wins;
    only when no candidate is local is the best-ranked one downloaded to its
    comfy-layout location under MODELS_PATH."""
    candidates = list(candidates)
    ordered = sorted(
        candidates,
        key=lambda c: (comfy_precision_rank(c, qtype=qtype), candidates.index(c)),
    )
    for repo_rel in ordered:
        found = resolve_comfy_file(
            comfy_local_rel(repo_rel), repo_id, local_only=True
        )
        if found is not None:
            return found
    if local_only:
        return None

    import huggingface_hub

    best = ordered[0]
    local_rel = comfy_local_rel(best)
    if status_fn is not None:
        status_fn(f"Downloading {best} from {repo_id} into {MODELS_PATH}")
    path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=best, token=hf_token, local_dir=MODELS_PATH
    )
    target = os.path.join(MODELS_PATH, local_rel)
    if os.path.abspath(path) != os.path.abspath(target):
        # move out of the packaging prefix into the shared comfy layout
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(path, target)
        return target
    return path


def find_file_recursive(root_dir: str, filename: str) -> Optional[str]:
    """First (breadth-stable, sorted) match of ``filename`` anywhere under
    ``root_dir``."""
    if not os.path.isdir(root_dir):
        return None
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def repo_id_from_name_or_path(
    name_or_path: Optional[str], default: str
) -> str:
    """Treat a hub-style ``name_or_path`` ("org/repo") as a replacement comfy
    repo; anything local (or an explicit .safetensors file) keeps the
    default."""
    if (
        name_or_path
        and not os.path.exists(name_or_path)
        and not name_or_path.endswith(".safetensors")
        and "/" in name_or_path
    ):
        return name_or_path
    return default


def resolve_comfy_file(
    rel_path: str,
    repo_id: str,
    override_path: Optional[str] = None,
    extra_roots: Optional[Iterable[str]] = None,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
    local_only: bool = False,
) -> Optional[str]:
    """Find a weight file at its local location, or download it there when
    (and only when) it is missing.

    Search order: ``override_path`` (must exist), the repo-relative path under
    MODELS_PATH (and each of ``extra_roots``), the bare filename at each root,
    any subfolder of the category folder (recursive — e.g.
    diffusion_models/my_custom_sub/), then the hub — downloaded to the
    repo-relative path under MODELS_PATH. With ``local_only`` the hub is never
    touched and a miss returns None.
    """
    if override_path is not None:
        if not os.path.exists(override_path):
            raise FileNotFoundError(
                f"Override path for {rel_path} does not exist: {override_path}"
            )
        return override_path

    filename = os.path.basename(rel_path)
    category = os.path.dirname(rel_path)
    roots = [MODELS_PATH] + [r for r in (extra_roots or []) if os.path.isdir(r)]
    for root in roots:
        for rel in (rel_path, filename):
            candidate = os.path.join(root, rel)
            if os.path.exists(candidate):
                return candidate
    for root in roots:
        found = find_file_recursive(os.path.join(root, category), filename)
        if found is not None:
            return found

    if local_only:
        return None

    import huggingface_hub

    if status_fn is not None:
        status_fn(f"Downloading {rel_path} from {repo_id} into {MODELS_PATH}")
    return huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=rel_path, token=hf_token, local_dir=MODELS_PATH
    )


def resolve_named_file(
    path: str,
    component: str = "model",
    hf_token: Optional[str] = None,
) -> str:
    """Resolve an explicit .safetensors reference: a local file, a file already
    under MODELS_PATH, or an 'org/repo/path/file.safetensors' hub path
    (downloaded into the models folder at its repo-relative path)."""
    if os.path.exists(path):
        return path
    splits = path.split("/")
    if len(splits) < 3:
        raise ValueError(
            f"Invalid {component} path: {path}. Must be a local file or "
            "'org/repo/filename.safetensors' to download from the Hugging Face Hub."
        )
    rel_path = "/".join(splits[2:])
    for candidate in (
        os.path.join(MODELS_PATH, rel_path),
        os.path.join(MODELS_PATH, splits[-1]),
    ):
        if os.path.exists(candidate):
            return candidate

    import huggingface_hub

    return huggingface_hub.hf_hub_download(
        repo_id="/".join(splits[:2]),
        filename=rel_path,
        token=hf_token,
        local_dir=MODELS_PATH,
    )
