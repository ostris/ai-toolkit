def convert_to_diffusers(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        # Base model name change
        if key.startswith("diffusion_model."):
            new_key = key.replace("diffusion_model.", "transformer.")

        # Attention blocks conversion
        if "self_attn" in new_key:
            new_key = new_key.replace("self_attn", "attn1")
        elif "cross_attn" in new_key:
            new_key = new_key.replace("cross_attn", "attn2")

        # Attention components conversion
        parts = new_key.split(".")
        for i, part in enumerate(parts):
            if part in ["q", "k", "v"]:
                parts[i] = f"to_{part}"
            elif part == "k_img":
                parts[i] = "add_k_proj"
            elif part == "v_img":
                parts[i] = "add_v_proj"
            elif part == "o":
                parts[i] = "to_out.0"
        new_key = ".".join(parts)

        # FFN conversion
        if "ffn.0" in new_key:
            new_key = new_key.replace("ffn.0", "ffn.net.0.proj")
        elif "ffn.2" in new_key:
            new_key = new_key.replace("ffn.2", "ffn.net.2")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def convert_to_original(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        # Base model name change
        if key.startswith("transformer."):
            new_key = key.replace("transformer.", "diffusion_model.")

        # Attention blocks conversion
        if "attn1" in new_key:
            new_key = new_key.replace("attn1", "self_attn")
        elif "attn2" in new_key:
            new_key = new_key.replace("attn2", "cross_attn")

        # Attention components conversion
        if "to_out.0" in new_key:
            new_key = new_key.replace("to_out.0", "o")
        elif "to_q" in new_key:
            new_key = new_key.replace("to_q", "q")
        elif "to_k" in new_key:
            new_key = new_key.replace("to_k", "k")
        elif "to_v" in new_key:
            new_key = new_key.replace("to_v", "v")
        
        # img attn projection
        elif "add_k_proj" in new_key:
            new_key = new_key.replace("add_k_proj", "k_img")
        elif "add_v_proj" in new_key:
            new_key = new_key.replace("add_v_proj", "v_img")

        # FFN conversion
        if "ffn.net.0.proj" in new_key:
            new_key = new_key.replace("ffn.net.0.proj", "ffn.0")
        elif "ffn.net.2" in new_key:
            new_key = new_key.replace("ffn.net.2", "ffn.2")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict
