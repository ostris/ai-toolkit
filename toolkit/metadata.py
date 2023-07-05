import json

software_meta = {
    "name": "ai-toolkit",
    "url": "https://github.com/ostris/ai-toolkit"
}


def create_meta(dict_list, name=None):
    meta = {}
    for d in dict_list:
        for key, value in d.items():
            meta[key] = value

    if "name" not in meta:
        meta["name"] = "[name]"

    meta["software"] = software_meta

    # convert to string to handle replacements
    meta_string = json.dumps(meta)
    if name is not None:
        meta_string = meta_string.replace("[name]", name)
    return json.loads(meta_string)


def prep_meta_for_safetensors(meta):
    # safetensors can only be one level deep
    for key, value in meta.items():
        if isinstance(value, dict):
            meta[key] = json.dumps(value)
    return meta
