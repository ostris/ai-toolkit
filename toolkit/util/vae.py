from diffusers import AutoencoderKL


def load_vae(vae_path, dtype):
    try:
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=dtype,
        )
    except Exception as e:
        try:
            vae = AutoencoderKL.from_pretrained(
                vae_path.vae_path,
                subfolder="vae",
                torch_dtype=dtype,
            )
        except Exception as e:
            raise ValueError(f"Failed to load VAE from {vae_path}: {e}")
    vae.to(dtype)
    return vae
