import huggingface_hub

SUPPORTED_MODELS = {
    "wd14-vit-v1": {
        "model": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger", "model.onnx",
            revision="213a7bd66d93407911b8217e806a95edc3593eed"
        ),
        "tags": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger", "selected_tags.csv",
            revision="213a7bd66d93407911b8217e806a95edc3593eed"
        ),
    },
    "wd14-vit-v2": {
        "model": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "model.onnx",
            revision="1f3f3e8ae769634e31e1ef696df11ec37493e4f2",
        ),
        "tags": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "selected_tags.csv",
            revision="1f3f3e8ae769634e31e1ef696df11ec37493e4f2",
        ),
    },
    # v1 & v2 are both using the same v2 model
    "wd14-swinv2-v1": {
        "model": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "model.onnx",
            revision="cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
        "tags": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "selected_tags.csv",
            revision="cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
    },
    "wd14-swinv2-v2": {
        "model": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "model.onnx",
            revision="cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
        "tags": lambda: huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "selected_tags.csv",
            revision="cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
    },
}


def get_model_and_labels(model: str):
    if model == "wd14-vit.v1":
        model = "wd14-vit-v1"
    elif model == "wd14-vit.v2":
        model = "wd14-vit-v2"

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model} is not supported. Supported models are: {list(SUPPORTED_MODELS.keys())}"
        )

    return SUPPORTED_MODELS[model]["model"](), SUPPORTED_MODELS[model]["tags"]()
