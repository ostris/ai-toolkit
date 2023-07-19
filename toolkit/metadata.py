import json
from collections import OrderedDict
from info import software_meta


def get_meta_for_safetensors(meta: OrderedDict, name=None) -> OrderedDict:
    # stringify the meta and reparse OrderedDict to replace [name] with name
    meta_string = json.dumps(meta)
    if name is not None:
        meta_string = meta_string.replace("[name]", name)
    save_meta = json.loads(meta_string, object_pairs_hook=OrderedDict)
    save_meta["software"] = software_meta
    # safetensors can only be one level deep
    for key, value in save_meta.items():
        # if not float, int, bool, or str, convert to json string
        if not isinstance(value, str):
            save_meta[key] = json.dumps(value)
    return save_meta


def parse_metadata_from_safetensors(meta: OrderedDict) -> OrderedDict:
    parsed_meta = OrderedDict()
    for key, value in meta.items():
        try:
            parsed_meta[key] = json.loads(value)
        except json.decoder.JSONDecodeError:
            parsed_meta[key] = value
    return meta
