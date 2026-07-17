import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


def test_frozen_dtype_keys_survive_save_load_and_dtype_conversion(tmp_path):
    cache_path = tmp_path / "prompt.safetensors"
    original = AdvancedPromptEmbeds(
        text_embeds=[torch.ones(2, dtype=torch.float32)],
        token_ids=[torch.tensor([1, 2], dtype=torch.int64)],
    )
    original.frozen_dtype_keys = ["token_ids"]

    original.save(str(cache_path))
    loaded = AdvancedPromptEmbeds.load(str(cache_path))
    converted = loaded.to(dtype=torch.bfloat16)

    assert loaded.frozen_dtype_keys == ["token_ids"]
    assert converted.text_embeds[0].dtype == torch.bfloat16
    assert converted.token_ids[0].dtype == torch.int64
