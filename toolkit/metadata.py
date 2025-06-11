import json
from collections import OrderedDict
from io import BytesIO

import safetensors
from safetensors import safe_open

from info import software_meta
from toolkit.train_tools import addnet_hash_legacy
from toolkit.train_tools import addnet_hash_safetensors


def get_meta_for_safetensors(meta: OrderedDict, name=None, add_software_info=True) -> OrderedDict:
    # stringify the meta and reparse OrderedDict to replace [name] with name
    meta_string = json.dumps(meta, ensure_ascii=False)
    if name is not None:
        meta_string = meta_string.replace("[name]", name)
    save_meta = json.loads(meta_string, object_pairs_hook=OrderedDict)
    if add_software_info:
        save_meta["software"] = software_meta
    # safetensors can only be one level deep
    for key, value in save_meta.items():
        # if not float, int, bool, or str, convert to json string
        if not isinstance(value, str):
            save_meta[key] = json.dumps(value, ensure_ascii=False)
    # add the pt format
    save_meta["format"] = "pt"
    return save_meta


def add_model_hash_to_meta(state_dict, meta: OrderedDict) -> OrderedDict:
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in meta.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(state_dict, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    meta["sshs_model_hash"] = model_hash
    meta["sshs_legacy_hash"] = legacy_hash
    return meta


def add_base_model_info_to_meta(
        meta: OrderedDict,
        base_model: str = None,
        is_v1: bool = False,
        is_v2: bool = False,
        is_xl: bool = False,
) -> OrderedDict:
    if base_model is not None:
        meta['ss_base_model'] = base_model
    elif is_v2:
        meta['ss_v2'] = True
        meta['ss_base_model_version'] = 'sd_2.1'

    elif is_xl:
        meta['ss_base_model_version'] = 'sdxl_1.0'
    else:
        # default to v1.5
        meta['ss_base_model_version'] = 'sd_1.5'
    return meta


def parse_metadata_from_safetensors(meta: OrderedDict) -> OrderedDict:
    parsed_meta = OrderedDict()
    for key, value in meta.items():
        try:
            parsed_meta[key] = json.loads(value)
        except json.decoder.JSONDecodeError:
            parsed_meta[key] = value
    return parsed_meta


def load_metadata_from_safetensors(file_path: str) -> OrderedDict:
    try:
        with safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
        return parse_metadata_from_safetensors(metadata)
    except Exception as e:
        print(f"Error loading metadata from {file_path}: {e}")
        return OrderedDict()
