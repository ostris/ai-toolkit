"""Strict artifact persistence for three-phase trigger-binding training.

This module is intentionally independent from trainer and model internals. It stores
one artifact class per safetensors file and treats every persisted manifest as an
untrusted input when loading.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ARTIFACT_SCHEMA = "ai-toolkit.trigger-binding-artifact"
ARTIFACT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "ai-toolkit.trigger-binding-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
ARTIFACT_TYPES = frozenset({"embedding", "te_adapter", "tap_adapter", "diffusion_lora"})
ARTIFACT_MANIFEST_SCHEMAS = {
    artifact_type: f"{ARTIFACT_SCHEMA}.{artifact_type}.v{ARTIFACT_SCHEMA_VERSION}"
    for artifact_type in ARTIFACT_TYPES
}

_METADATA_SCHEMA = "trigger_binding.schema"
_METADATA_VERSION = "trigger_binding.schema_version"
_METADATA_TYPE = "trigger_binding.artifact_type"
_METADATA_MANIFEST = "trigger_binding.manifest"
_METADATA_MANIFEST_SHA256 = "trigger_binding.manifest_sha256"
_REQUIRED_METADATA_KEYS = frozenset(
    {
        _METADATA_SCHEMA,
        _METADATA_VERSION,
        _METADATA_TYPE,
        _METADATA_MANIFEST,
        _METADATA_MANIFEST_SHA256,
    }
)

PathLike = Union[str, os.PathLike]


class ArtifactValidationError(ValueError):
    """Raised when an artifact or checkpoint fails closed validation."""


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically for hashing and metadata."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json_dumps(to_json_compatible(value)).encode("utf-8"))


def phase_fingerprint(phase: Any) -> str:
    return fingerprint(phase)


def source_fingerprint(source: Any) -> str:
    return fingerprint(source)


def config_fingerprint(config: Any) -> str:
    return fingerprint(config)


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _object_config(value: Any) -> Dict[str, Any]:
    fields = {
        key: field_value
        for key, field_value in vars(value).items()
        if not key.startswith("_") and not callable(field_value)
    }
    return {"__type__": "object", "class": _qualified_name(value), "fields": to_json_compatible(fields)}


def to_json_compatible(value: Any) -> Any:
    """Encode checkpoint state, including Python/NumPy/Torch RNG state, as JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Non-finite floats are not JSON compatible")
        return value
    if isinstance(value, bytes):
        return {"__type__": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, bytearray):
        return {"__type__": "bytearray", "base64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [to_json_compatible(item) for item in value]}
    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON-compatible mappings require string keys")
        return {key: to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, torch.dtype):
        return {"__type__": "torch.dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = _tensor_bytes(tensor)
        return {
            "__type__": "torch.Tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "base64": base64.b64encode(raw).decode("ascii"),
        }
    module = type(value).__module__
    if module.startswith("numpy"):
        import numpy as np

        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {
                "__type__": "numpy.ndarray",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "base64": base64.b64encode(array.tobytes()).decode("ascii"),
            }
        if isinstance(value, np.generic):
            return to_json_compatible(value.item())
    if hasattr(value, "__dict__"):
        return _object_config(value)
    raise TypeError(f"Unsupported JSON-compatible value: {_qualified_name(value)}")


def from_json_compatible(value: Any) -> Any:
    """Decode values produced by :func:`to_json_compatible`."""
    if isinstance(value, list):
        return [from_json_compatible(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("__type__")
    if tag is None:
        return {key: from_json_compatible(item) for key, item in value.items()}
    if tag in {"bytes", "bytearray"}:
        decoded = base64.b64decode(value["base64"], validate=True)
        return decoded if tag == "bytes" else bytearray(decoded)
    if tag == "tuple":
        return tuple(from_json_compatible(item) for item in value["items"])
    if tag == "torch.dtype":
        dtype = getattr(torch, value["value"], None)
        if not isinstance(dtype, torch.dtype):
            raise ArtifactValidationError(f"Unknown torch dtype: {value['value']}")
        return dtype
    if tag == "torch.Tensor":
        dtype = getattr(torch, value["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ArtifactValidationError(f"Unknown torch dtype: {value['dtype']}")
        raw = base64.b64decode(value["base64"], validate=True)
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        expected_numel = math.prod(value["shape"])
        if tensor.numel() != expected_numel:
            raise ArtifactValidationError("Encoded tensor byte count does not match its shape")
        return tensor.reshape(value["shape"])
    if tag == "numpy.ndarray":
        import numpy as np

        raw = base64.b64decode(value["base64"], validate=True)
        array = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).copy()
        expected_size = math.prod(value["shape"])
        if array.size != expected_size:
            raise ArtifactValidationError("Encoded NumPy byte count does not match its shape")
        return array.reshape(value["shape"])
    if tag == "object":
        return {"class": value["class"], "fields": from_json_compatible(value["fields"])}
    raise ArtifactValidationError(f"Unknown JSON compatibility tag: {tag}")


def encode_rng_state(rng_state: Any) -> Any:
    return to_json_compatible(rng_state)


def decode_rng_state(encoded: Any) -> Any:
    return from_json_compatible(encoded)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    normalized = tensor.detach().cpu().contiguous()
    if normalized.numel() == 0:
        return b""
    # PyTorch cannot reinterpret a zero-dimensional tensor as a dtype with a
    # different element size. Flattening preserves the exact storage bytes and
    # also handles BF16, scalar adapter scales, and non-contiguous inputs.
    return normalized.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(_tensor_bytes(tensor))


def _validate_tensor_mapping(tensors: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping) or not tensors:
        raise ArtifactValidationError("Artifact tensors must be a non-empty mapping")
    normalized: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if not isinstance(key, str) or not key:
            raise ArtifactValidationError("Artifact tensor keys must be non-empty strings")
        if key in normalized:
            raise ArtifactValidationError(f"Duplicate tensor key: {key}")
        if not isinstance(tensor, torch.Tensor):
            raise ArtifactValidationError(f"Artifact value for {key!r} is not a tensor")
        if tensor.layout != torch.strided:
            raise ArtifactValidationError(f"Artifact tensor {key!r} must use strided layout")
        normalized[key] = tensor.detach().cpu().contiguous()
    return normalized


def build_artifact_manifest(
    artifact_type: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    phase: Any,
    source: Any,
    config: Any,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if artifact_type not in ARTIFACT_TYPES:
        raise ArtifactValidationError(f"Unsupported artifact type: {artifact_type!r}")
    normalized = _validate_tensor_mapping(tensors)
    specs = {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": tensor_sha256(tensor),
        }
        for key, tensor in sorted(normalized.items())
    }
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "artifact_schema": ARTIFACT_MANIFEST_SCHEMAS[artifact_type],
        "phase_fingerprint": phase_fingerprint(phase),
        "source_fingerprint": source_fingerprint(source),
        "config_fingerprint": config_fingerprint(config),
        "tensors": specs,
    }
    if extra is not None:
        manifest["extra"] = to_json_compatible(dict(extra))
    return manifest


def _artifact_metadata(manifest: Mapping[str, Any]) -> Dict[str, str]:
    manifest_json = canonical_json_dumps(manifest)
    return {
        _METADATA_SCHEMA: ARTIFACT_SCHEMA,
        _METADATA_VERSION: str(ARTIFACT_SCHEMA_VERSION),
        _METADATA_TYPE: str(manifest["artifact_type"]),
        _METADATA_MANIFEST: manifest_json,
        _METADATA_MANIFEST_SHA256: sha256_bytes(manifest_json.encode("utf-8")),
    }


def _atomic_replace(temp_path: str, destination: Path) -> None:
    try:
        with open(temp_path, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        # Some Windows filesystems reject fsync on safetensors temp handles.
        # os.replace remains atomic on the same volume.
        pass
    os.replace(temp_path, destination)
    try:
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save_artifact(
    path: PathLike,
    artifact_type: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    phase: Any,
    source: Any,
    config: Any,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically save a typed safetensors artifact and return its manifest."""
    destination = Path(path)
    if destination.suffix.lower() != ".safetensors":
        raise ArtifactValidationError("Artifact path must end with .safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = _validate_tensor_mapping(tensors)
    manifest = build_artifact_manifest(
        artifact_type,
        normalized,
        phase=phase,
        source=source,
        config=config,
        extra=extra,
    )
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(file_descriptor)
    try:
        save_file(normalized, temp_path, metadata=_artifact_metadata(manifest))
        _atomic_replace(temp_path, destination)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise
    return manifest


def _parse_artifact_manifest(metadata: Optional[Mapping[str, str]]) -> Dict[str, Any]:
    if metadata is None or not _REQUIRED_METADATA_KEYS.issubset(metadata):
        missing = sorted(_REQUIRED_METADATA_KEYS.difference(metadata or {}))
        raise ArtifactValidationError(f"Missing required safetensors metadata: {missing}")
    if metadata[_METADATA_SCHEMA] != ARTIFACT_SCHEMA:
        raise ArtifactValidationError("Artifact metadata schema mismatch")
    if metadata[_METADATA_VERSION] != str(ARTIFACT_SCHEMA_VERSION):
        raise ArtifactValidationError("Unsupported artifact metadata schema version")
    manifest_json = metadata[_METADATA_MANIFEST]
    if sha256_bytes(manifest_json.encode("utf-8")) != metadata[_METADATA_MANIFEST_SHA256]:
        raise ArtifactValidationError("Artifact manifest metadata hash mismatch")
    try:
        manifest = json.loads(manifest_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Artifact manifest is not valid JSON") from exc
    if canonical_json_dumps(manifest) != manifest_json:
        raise ArtifactValidationError("Artifact manifest is not canonically encoded")
    required = {
        "schema",
        "schema_version",
        "artifact_type",
        "artifact_schema",
        "phase_fingerprint",
        "source_fingerprint",
        "config_fingerprint",
        "tensors",
    }
    if set(manifest).difference(required | {"extra"}) or not required.issubset(manifest):
        raise ArtifactValidationError("Artifact manifest has missing or unknown top-level keys")
    if manifest["schema"] != ARTIFACT_SCHEMA or manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("Artifact manifest schema mismatch")
    if manifest["artifact_type"] not in ARTIFACT_TYPES:
        raise ArtifactValidationError("Artifact manifest type is unsupported")
    if manifest["artifact_schema"] != ARTIFACT_MANIFEST_SCHEMAS[manifest["artifact_type"]]:
        raise ArtifactValidationError("Artifact-specific manifest schema mismatch")
    if metadata[_METADATA_TYPE] != manifest["artifact_type"]:
        raise ArtifactValidationError("Artifact type metadata disagrees with manifest")
    for name in ("phase_fingerprint", "source_fingerprint", "config_fingerprint"):
        value = manifest[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ArtifactValidationError(f"Invalid {name}")
    if not isinstance(manifest["tensors"], dict) or not manifest["tensors"]:
        raise ArtifactValidationError("Artifact manifest tensor map is empty")
    return manifest


def load_artifact(
    path: PathLike,
    *,
    expected_type: Optional[str] = None,
    expected_keys: Optional[Sequence[str]] = None,
    expected_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    expected_phase_fingerprint: Optional[str] = None,
    expected_source_fingerprint: Optional[str] = None,
    expected_config_fingerprint: Optional[str] = None,
    expected_file_sha256: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load an artifact only after exact key, shape, dtype and hash validation."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ArtifactValidationError(f"Artifact file does not exist: {artifact_path}")
    if expected_file_sha256 is not None and sha256_file(artifact_path) != expected_file_sha256:
        raise ArtifactValidationError("Artifact file SHA-256 mismatch")
    try:
        with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
            manifest = _parse_artifact_manifest(handle.metadata())
            file_keys = set(handle.keys())
            manifest_keys = set(manifest["tensors"])
            if file_keys != manifest_keys:
                raise ArtifactValidationError(
                    f"Artifact tensor keys mismatch: file={sorted(file_keys)}, manifest={sorted(manifest_keys)}"
                )
            tensors = {key: handle.get_tensor(key) for key in sorted(file_keys)}
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"Unable to read safetensors artifact: {artifact_path}") from exc

    if expected_type is not None and manifest["artifact_type"] != expected_type:
        raise ArtifactValidationError("Artifact type does not match the expected type")
    expected_fingerprints = {
        "phase_fingerprint": expected_phase_fingerprint,
        "source_fingerprint": expected_source_fingerprint,
        "config_fingerprint": expected_config_fingerprint,
    }
    for name, expected in expected_fingerprints.items():
        if expected is not None and manifest[name] != expected:
            raise ArtifactValidationError(f"Artifact {name} mismatch")
    if expected_keys is not None and set(expected_keys) != set(tensors):
        raise ArtifactValidationError("Artifact keys do not match expected keys")
    if expected_shapes is not None:
        if set(expected_shapes) != set(tensors):
            raise ArtifactValidationError("Expected shape keys do not match artifact keys")
        for key, shape in expected_shapes.items():
            if list(tensors[key].shape) != list(shape):
                raise ArtifactValidationError(f"Artifact tensor {key!r} has unexpected shape")

    for key, tensor in tensors.items():
        spec = manifest["tensors"].get(key)
        if not isinstance(spec, dict) or set(spec) != {"shape", "dtype", "sha256"}:
            raise ArtifactValidationError(f"Invalid tensor manifest entry for {key!r}")
        if list(tensor.shape) != spec["shape"]:
            raise ArtifactValidationError(f"Artifact tensor {key!r} shape mismatch")
        if str(tensor.dtype).removeprefix("torch.") != spec["dtype"]:
            raise ArtifactValidationError(f"Artifact tensor {key!r} dtype mismatch")
        if tensor_sha256(tensor) != spec["sha256"]:
            raise ArtifactValidationError(f"Artifact tensor {key!r} SHA-256 mismatch")
    return {key: tensor.to(device) for key, tensor in tensors.items()}, manifest


def artifact_reference(path: PathLike, *, relative_to: Optional[PathLike] = None) -> Dict[str, Any]:
    artifact_path = Path(path)
    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        manifest = _parse_artifact_manifest(handle.metadata())
    reference_path = os.path.relpath(artifact_path, relative_to) if relative_to is not None else str(artifact_path)
    return {
        "path": reference_path.replace(os.sep, "/"),
        "artifact_type": manifest["artifact_type"],
        "sha256": sha256_file(artifact_path),
        "manifest_sha256": sha256_bytes(canonical_json_dumps(manifest).encode("utf-8")),
    }


def build_checkpoint_manifest(
    *,
    phase: Any,
    source: Any,
    config: Any,
    step: int,
    artifacts: Mapping[str, Mapping[str, Any]],
    rng_state: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ArtifactValidationError("Checkpoint step must be a non-negative integer")
    if not isinstance(artifacts, Mapping):
        raise ArtifactValidationError("Checkpoint artifacts must be a mapping")
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "phase_fingerprint": phase_fingerprint(phase),
        "source_fingerprint": source_fingerprint(source),
        "config_fingerprint": config_fingerprint(config),
        "artifacts": to_json_compatible(dict(artifacts)),
        "rng_state": encode_rng_state(rng_state),
    }
    if extra is not None:
        manifest["extra"] = to_json_compatible(dict(extra))
    validate_checkpoint_manifest(manifest)
    return manifest


def validate_checkpoint_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "schema_version",
        "step",
        "phase_fingerprint",
        "source_fingerprint",
        "config_fingerprint",
        "artifacts",
        "rng_state",
    }
    if not isinstance(manifest, Mapping):
        raise ArtifactValidationError("Checkpoint manifest must be a mapping")
    if set(manifest).difference(required | {"extra"}) or not required.issubset(manifest):
        raise ArtifactValidationError("Checkpoint manifest has missing or unknown top-level keys")
    if manifest["schema"] != CHECKPOINT_SCHEMA or manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactValidationError("Checkpoint manifest schema mismatch")
    if not isinstance(manifest["step"], int) or isinstance(manifest["step"], bool) or manifest["step"] < 0:
        raise ArtifactValidationError("Checkpoint manifest step is invalid")
    for name in ("phase_fingerprint", "source_fingerprint", "config_fingerprint"):
        value = manifest[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ArtifactValidationError(f"Checkpoint {name} is invalid")
    if not isinstance(manifest["artifacts"], dict):
        raise ArtifactValidationError("Checkpoint artifact references must be a mapping")
    for name, reference in manifest["artifacts"].items():
        if not isinstance(name, str) or not isinstance(reference, dict):
            raise ArtifactValidationError("Checkpoint artifact reference is invalid")
        if set(reference) != {"path", "artifact_type", "sha256", "manifest_sha256"}:
            raise ArtifactValidationError(f"Checkpoint artifact reference {name!r} has invalid keys")
        if reference["artifact_type"] not in ARTIFACT_TYPES:
            raise ArtifactValidationError(f"Checkpoint artifact reference {name!r} has invalid type")
        if not all(isinstance(reference[key], str) for key in reference):
            raise ArtifactValidationError(f"Checkpoint artifact reference {name!r} contains non-string values")
        if len(reference["sha256"]) != 64 or len(reference["manifest_sha256"]) != 64:
            raise ArtifactValidationError(f"Checkpoint artifact reference {name!r} has invalid hash")
    try:
        decode_rng_state(manifest["rng_state"])
    except Exception as exc:
        raise ArtifactValidationError("Checkpoint RNG state encoding is invalid") from exc


def save_checkpoint_manifest(path: PathLike, manifest: Mapping[str, Any]) -> None:
    validate_checkpoint_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json_dumps(manifest) + "\n").encode("utf-8")
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint_manifest(
    path: PathLike,
    *,
    verify_artifacts: bool = True,
    expected_phase_fingerprint: Optional[str] = None,
    expected_source_fingerprint: Optional[str] = None,
    expected_config_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = Path(path)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        raise ArtifactValidationError(f"Unable to read checkpoint manifest: {manifest_path}") from exc
    validate_checkpoint_manifest(manifest)
    expected_fingerprints = {
        "phase_fingerprint": expected_phase_fingerprint,
        "source_fingerprint": expected_source_fingerprint,
        "config_fingerprint": expected_config_fingerprint,
    }
    for name, expected in expected_fingerprints.items():
        if expected is not None and manifest[name] != expected:
            raise ArtifactValidationError(f"Checkpoint {name} mismatch")
    if verify_artifacts:
        for name, reference in manifest["artifacts"].items():
            artifact_path = Path(reference["path"])
            if not artifact_path.is_absolute():
                artifact_path = manifest_path.parent / artifact_path
            _, artifact_manifest = load_artifact(
                artifact_path,
                expected_type=reference["artifact_type"],
                expected_file_sha256=reference["sha256"],
            )
            actual_manifest_hash = sha256_bytes(canonical_json_dumps(artifact_manifest).encode("utf-8"))
            if actual_manifest_hash != reference["manifest_sha256"]:
                raise ArtifactValidationError(f"Checkpoint artifact {name!r} manifest hash mismatch")
    return deepcopy(manifest)
