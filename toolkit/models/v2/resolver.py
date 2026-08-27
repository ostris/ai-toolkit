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
