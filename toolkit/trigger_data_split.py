import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union


SPLIT_MANIFEST_SCHEMA = 'ai-toolkit.trigger-data-split'
SPLIT_MANIFEST_SCHEMA_VERSION = 1
SplitName = str
ItemIdGetter = Callable[[Any], str]


def normalize_dataset_relative_item_id(item_id: Union[str, os.PathLike]) -> str:
    """Return a platform-independent, dataset-relative item identifier."""
    if not isinstance(item_id, (str, os.PathLike)):
        raise ValueError('dataset-relative item ID must be a string or path-like value')
    value = os.fspath(item_id).strip().replace('\\', '/')
    if not value:
        raise ValueError('dataset-relative item ID must not be empty')
    if value.startswith('/') or (len(value) >= 2 and value[1] == ':'):
        raise ValueError(f'dataset-relative item ID must not be absolute: {item_id}')

    parts = []
    for part in value.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            raise ValueError(f'dataset-relative item ID must not escape the dataset: {item_id}')
        parts.append(part)
    if not parts:
        raise ValueError('dataset-relative item ID must identify an item')
    return PurePosixPath(*parts).as_posix()


def dataset_relative_item_id(path: Union[str, os.PathLike], dataset_root: Union[str, os.PathLike]) -> str:
    root = os.path.abspath(os.path.expanduser(os.fspath(dataset_root)))
    absolute_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    try:
        common = os.path.commonpath((root, absolute_path))
    except ValueError as error:
        raise ValueError(f'item is not inside dataset root: {path}') from error
    if os.path.normcase(common) != os.path.normcase(root):
        raise ValueError(f'item is not inside dataset root: {path}')
    return normalize_dataset_relative_item_id(os.path.relpath(absolute_path, root))


def paired_caption_item_id(
    caption_path: Union[str, os.PathLike],
    dataset_root: Union[str, os.PathLike],
    image_item_ids: Iterable[str],
) -> str:
    """Resolve a sidecar caption to its image ID by dataset-relative stem."""
    caption_id = dataset_relative_item_id(caption_path, dataset_root)
    caption_stem = str(PurePosixPath(caption_id).with_suffix(''))
    matches = [
        item_id
        for item_id in normalize_item_ids(image_item_ids)
        if str(PurePosixPath(item_id).with_suffix('')) == caption_stem
    ]
    if len(matches) != 1:
        raise ValueError(
            f'caption must resolve to exactly one image item ID; found {len(matches)} for {caption_id}'
        )
    return matches[0]


def _default_item_id_getter(item: Any) -> str:
    if isinstance(item, (str, os.PathLike)):
        return os.fspath(item)
    if isinstance(item, Mapping):
        for key in ('dataset_relative_item_id', 'item_id'):
            if key in item:
                return item[key]
    for attribute in ('dataset_relative_item_id', 'item_id'):
        value = getattr(item, attribute, None)
        if value:
            return value
    raise ValueError('item does not expose dataset_relative_item_id or item_id')


def normalize_item_ids(item_ids: Iterable[Union[str, os.PathLike]]) -> List[str]:
    normalized = [normalize_dataset_relative_item_id(item_id) for item_id in item_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError('dataset-relative item IDs must be unique after normalization')
    return sorted(normalized)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def _sha256(payload: Union[bytes, Mapping[str, Any]]) -> str:
    encoded = payload if isinstance(payload, bytes) else _canonical_json(payload)
    return hashlib.sha256(encoded).hexdigest()


def compute_dataset_fingerprint(item_ids: Iterable[Union[str, os.PathLike]]) -> str:
    return _sha256({'item_ids': normalize_item_ids(item_ids)})


def heldout_item_count(item_count: int, heldout_fraction: float) -> int:
    """Round half up, then clamp so train and held-out are both non-empty."""
    if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count < 2:
        raise ValueError('data split requires at least two unique image item IDs')
    if not math.isfinite(heldout_fraction) or not 0.0 < heldout_fraction < 1.0:
        raise ValueError('heldout_fraction must be finite and strictly between 0 and 1')
    rounded = int(math.floor(item_count * heldout_fraction + 0.5))
    return min(item_count - 1, max(1, rounded))


@dataclass(frozen=True)
class TriggerDataSplitManifest:
    schema: str
    schema_version: int
    seed: int
    heldout_fraction: float
    dataset_fingerprint: str
    train_item_ids: Tuple[str, ...]
    heldout_item_ids: Tuple[str, ...]
    split_hash: str

    @property
    def train_ids(self) -> Tuple[str, ...]:
        return self.train_item_ids

    @property
    def heldout_ids(self) -> Tuple[str, ...]:
        return self.heldout_item_ids

    def as_dict(self) -> Dict[str, Any]:
        return {
            'schema': self.schema,
            'schema_version': self.schema_version,
            'seed': self.seed,
            'heldout_fraction': self.heldout_fraction,
            'dataset_fingerprint': self.dataset_fingerprint,
            'train_item_ids': list(self.train_item_ids),
            'heldout_item_ids': list(self.heldout_item_ids),
            'split_hash': self.split_hash,
        }


def _manifest_hash_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            'schema',
            'schema_version',
            'seed',
            'heldout_fraction',
            'dataset_fingerprint',
            'train_item_ids',
            'heldout_item_ids',
        )
    }


def create_data_split_manifest(
    item_ids: Iterable[Union[str, os.PathLike]],
    *,
    seed: int,
    heldout_fraction: float = 0.1,
) -> TriggerDataSplitManifest:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('data split seed must be a non-negative integer')
    normalized_ids = normalize_item_ids(item_ids)
    count = heldout_item_count(len(normalized_ids), heldout_fraction)

    shuffled = list(normalized_ids)
    random.Random(seed).shuffle(shuffled)
    heldout_set = set(shuffled[:count])
    train_ids = tuple(item_id for item_id in normalized_ids if item_id not in heldout_set)
    heldout_ids = tuple(item_id for item_id in normalized_ids if item_id in heldout_set)
    payload = {
        'schema': SPLIT_MANIFEST_SCHEMA,
        'schema_version': SPLIT_MANIFEST_SCHEMA_VERSION,
        'seed': seed,
        'heldout_fraction': float(heldout_fraction),
        'dataset_fingerprint': compute_dataset_fingerprint(normalized_ids),
        'train_item_ids': list(train_ids),
        'heldout_item_ids': list(heldout_ids),
    }
    return TriggerDataSplitManifest(
        schema=payload['schema'],
        schema_version=payload['schema_version'],
        seed=payload['seed'],
        heldout_fraction=payload['heldout_fraction'],
        dataset_fingerprint=payload['dataset_fingerprint'],
        train_item_ids=train_ids,
        heldout_item_ids=heldout_ids,
        split_hash=_sha256(payload),
    )


def validate_data_split_manifest(
    manifest: Union[TriggerDataSplitManifest, Mapping[str, Any]],
    *,
    item_ids: Optional[Iterable[Union[str, os.PathLike]]] = None,
    expected_seed: Optional[int] = None,
    expected_heldout_fraction: Optional[float] = None,
) -> TriggerDataSplitManifest:
    payload = manifest.as_dict() if isinstance(manifest, TriggerDataSplitManifest) else dict(manifest)
    required = {
        'schema', 'schema_version', 'seed', 'heldout_fraction', 'dataset_fingerprint',
        'train_item_ids', 'heldout_item_ids', 'split_hash',
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f'data split manifest is missing fields: {missing}')
    if payload['schema'] != SPLIT_MANIFEST_SCHEMA or payload['schema_version'] != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError('unsupported data split manifest schema')

    train_ids = normalize_item_ids(payload['train_item_ids'])
    heldout_ids = normalize_item_ids(payload['heldout_item_ids'])
    assert_no_split_leakage(train_ids, heldout_ids)
    if not train_ids or not heldout_ids:
        raise ValueError('data split manifest must keep train and held-out sides non-empty')

    combined_ids = sorted(train_ids + heldout_ids)
    computed_fingerprint = compute_dataset_fingerprint(combined_ids)
    if payload['dataset_fingerprint'] != computed_fingerprint:
        raise ValueError('data split manifest dataset fingerprint is invalid')
    expected_hash = _sha256(_manifest_hash_payload({
        **payload,
        'train_item_ids': train_ids,
        'heldout_item_ids': heldout_ids,
    }))
    if payload['split_hash'] != expected_hash:
        raise ValueError('data split manifest hash is invalid')
    if expected_seed is not None and payload['seed'] != expected_seed:
        raise ValueError('data split manifest seed does not match configured seed')
    if expected_heldout_fraction is not None and not math.isclose(
        float(payload['heldout_fraction']), float(expected_heldout_fraction), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError('data split manifest heldout_fraction does not match configuration')
    if item_ids is not None:
        current_fingerprint = compute_dataset_fingerprint(item_ids)
        if current_fingerprint != payload['dataset_fingerprint']:
            raise ValueError('data split dataset fingerprint mismatch; refusing to reuse stale manifest')

    return TriggerDataSplitManifest(
        schema=payload['schema'],
        schema_version=int(payload['schema_version']),
        seed=int(payload['seed']),
        heldout_fraction=float(payload['heldout_fraction']),
        dataset_fingerprint=payload['dataset_fingerprint'],
        train_item_ids=tuple(train_ids),
        heldout_item_ids=tuple(heldout_ids),
        split_hash=payload['split_hash'],
    )


def persist_data_split_manifest(
    manifest: Union[TriggerDataSplitManifest, Mapping[str, Any]],
    path: Union[str, os.PathLike],
) -> str:
    validated = validate_data_split_manifest(manifest)
    destination = os.path.abspath(os.path.expanduser(os.fspath(path)))
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(destination)}.', suffix='.tmp', dir=parent or None
    )
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(validated.as_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_data_split_manifest(
    path: Union[str, os.PathLike],
    *,
    item_ids: Optional[Iterable[Union[str, os.PathLike]]] = None,
    expected_seed: Optional[int] = None,
    expected_heldout_fraction: Optional[float] = None,
) -> TriggerDataSplitManifest:
    source = os.path.abspath(os.path.expanduser(os.fspath(path)))
    with open(source, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError('data split manifest root must be a JSON object')
    return validate_data_split_manifest(
        payload,
        item_ids=item_ids,
        expected_seed=expected_seed,
        expected_heldout_fraction=expected_heldout_fraction,
    )


def get_or_create_data_split_manifest(
    item_ids: Iterable[Union[str, os.PathLike]],
    path: Union[str, os.PathLike],
    *,
    seed: int,
    heldout_fraction: float = 0.1,
    reuse_existing: bool = True,
) -> TriggerDataSplitManifest:
    normalized_ids = normalize_item_ids(item_ids)
    manifest_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.path.exists(manifest_path) and reuse_existing:
        return load_data_split_manifest(
            manifest_path,
            item_ids=normalized_ids,
            expected_seed=seed,
            expected_heldout_fraction=heldout_fraction,
        )
    manifest = create_data_split_manifest(
        normalized_ids,
        seed=seed,
        heldout_fraction=heldout_fraction,
    )
    persist_data_split_manifest(manifest, manifest_path)
    return manifest


def split_allowlist(
    manifest: Union[TriggerDataSplitManifest, Mapping[str, Any]],
    split: SplitName,
) -> Set[str]:
    validated = validate_data_split_manifest(manifest)
    if split in ('train', 'train_probe'):
        return set(validated.train_item_ids)
    if split in ('heldout', 'holdout', 'validation'):
        return set(validated.heldout_item_ids)
    raise ValueError("split must be 'train' or 'heldout'")


def item_is_allowed(
    item: Any,
    allowlist: Iterable[str],
    *,
    item_id_getter: ItemIdGetter = _default_item_id_getter,
) -> bool:
    normalized_allowlist = {
        normalize_dataset_relative_item_id(item_id) for item_id in allowlist
    }
    return normalize_dataset_relative_item_id(item_id_getter(item)) in normalized_allowlist


def filter_items_by_allowlist(
    items: Iterable[Any],
    allowlist: Iterable[str],
    *,
    item_id_getter: ItemIdGetter = _default_item_id_getter,
    require_all_allowlisted: bool = False,
) -> List[Any]:
    normalized_allowlist = {
        normalize_dataset_relative_item_id(item_id) for item_id in allowlist
    }
    filtered = []
    seen = set()
    for item in items:
        item_id = normalize_dataset_relative_item_id(item_id_getter(item))
        if item_id in normalized_allowlist:
            filtered.append(item)
            seen.add(item_id)
    if require_all_allowlisted and seen != normalized_allowlist:
        missing = sorted(normalized_allowlist - seen)
        raise ValueError(f'allowlisted item IDs were not found: {missing}')
    return filtered


def filter_items_for_split(
    items: Iterable[Any],
    manifest: Union[TriggerDataSplitManifest, Mapping[str, Any]],
    split: SplitName,
    *,
    item_id_getter: ItemIdGetter = _default_item_id_getter,
    require_all_allowlisted: bool = False,
) -> List[Any]:
    return filter_items_by_allowlist(
        items,
        split_allowlist(manifest, split),
        item_id_getter=item_id_getter,
        require_all_allowlisted=require_all_allowlisted,
    )


def assert_no_split_leakage(
    train_item_ids: Iterable[Union[str, os.PathLike]],
    heldout_item_ids: Iterable[Union[str, os.PathLike]],
) -> None:
    train = {normalize_dataset_relative_item_id(item_id) for item_id in train_item_ids}
    heldout = {normalize_dataset_relative_item_id(item_id) for item_id in heldout_item_ids}
    overlap = sorted(train & heldout)
    if overlap:
        raise ValueError(f'train/held-out leakage detected for item IDs: {overlap}')


# Concise aliases for integration call sites.
create_split_manifest = create_data_split_manifest
load_split_manifest = load_data_split_manifest
persist_split_manifest = persist_data_split_manifest
resolve_split_manifest = get_or_create_data_split_manifest
