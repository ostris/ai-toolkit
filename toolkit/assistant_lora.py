from typing import TYPE_CHECKING
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from safetensors.torch import load_file

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion


def load_assistant_lora_from_path(adapter_path, sd: 'StableDiffusion') -> LoRASpecialNetwork:
    if not sd.is_flux:
        raise ValueError("Only Flux models can load assistant adapters currently.")
    pipe = sd.pipeline
    print(f"Loading assistant adapter from {adapter_path}")
    adapter_name = adapter_path.split("/")[-1].split(".")[0]
    lora_state_dict = load_file(adapter_path)

    linear_dim = int(lora_state_dict['transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight'].shape[0])
    # linear_alpha = int(lora_state_dict['lora_transformer_single_transformer_blocks_0_attn_to_k.alpha'].item())
    linear_alpha = linear_dim
    transformer_only = 'transformer.proj_out.alpha' not in lora_state_dict
    # get dim and scale
    network_config = NetworkConfig(
        linear=linear_dim,
        linear_alpha=linear_alpha,
        transformer_only=transformer_only,
    )

    network = LoRASpecialNetwork(
        text_encoder=pipe.text_encoder,
        unet=pipe.transformer,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        is_flux=True,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_assistant_adapter=True,
        base_model=sd
    )
    network.apply_to(
        pipe.text_encoder,
        pipe.transformer,
        apply_text_encoder=False,
        apply_unet=True
    )
    network.force_to(sd.device_torch, dtype=sd.torch_dtype)
    network.eval()
    network._update_torch_multiplier()
    network.load_weights(lora_state_dict)
    network.is_active = True

    return network
