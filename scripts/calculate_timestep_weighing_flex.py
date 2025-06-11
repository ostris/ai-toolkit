import gc
import os, sys
from tqdm import tqdm
import numpy as np
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# set visible devices to 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# protect from formatting
if True:
    import torch
    from optimum.quanto import freeze, qfloat8, QTensor, qint4
    from diffusers import FluxTransformer2DModel, FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from toolkit.util.quantize import quantize, get_qtype
    from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModel, CLIPTokenizer
    from torchvision import transforms

qtype = "qfloat8"
dtype = torch.bfloat16
# base_model_path = "black-forest-labs/FLUX.1-dev"
base_model_path = "ostris/Flex.1-alpha"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loading Transformer...")
prompt = "Photo of a man and a woman in a park, sunny day"

output_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
output_path = os.path.join(output_root, "flex_timestep_weights.json")
img_output_path = os.path.join(output_root, "flex_timestep_weights.png")

quantization_type = get_qtype(qtype)

def flush():
    torch.cuda.empty_cache()
    gc.collect()
    
pil_to_tensor = transforms.ToTensor()
    
with torch.no_grad():
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path,
        subfolder='transformer',
        torch_dtype=dtype
    )

    transformer.to(device, dtype=dtype)

    print("Quantizing Transformer...")
    quantize(transformer, weights=quantization_type)
    freeze(transformer)
    flush()

    print("Loading Scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    print("Loading Autoencoder...")
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)

    vae.to(device, dtype=dtype)

    flush()
    print("Loading Text Encoder...")
    tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_2", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder_2", torch_dtype=dtype)
    text_encoder_2.to(device, dtype=dtype)

    print("Quantizing Text Encoder...")
    quantize(text_encoder_2, weights=get_qtype(qtype))
    freeze(text_encoder_2)
    flush()

    print("Loading CLIP")
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder.to(device, dtype=dtype)

    print("Making pipe")
                
    pipe: FluxPipeline = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,
    )
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer = transformer
    
    pipe.to(device, dtype=dtype)
    
    print("Encoding prompt...")
    
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt,
        prompt_2=prompt,
        device=device
    )
    
    
    generator = torch.manual_seed(42)
    
    height = 1024
    width = 1024
    
    print("Generating image...")
    
    # Fix a bug in diffusers/torch
    def callback_on_step_end(pipe, i, t, callback_kwargs):
        latents = callback_kwargs["latents"]
        if latents.dtype != dtype:
            latents = latents.to(dtype)
        return {"latents": latents}
    img = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=height,
        width=height,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator,
        callback_on_step_end=callback_on_step_end,
    ).images[0]
    
    img.save(img_output_path)
    print(f"Image saved to {img_output_path}")
    
    print("Encoding image...")
    # img is a PIL image. convert it to a -1 to 1 tensor
    img = pil_to_tensor(img)
    img = img.unsqueeze(0)  # add batch dimension
    img = img * 2 - 1  # convert to -1 to 1 range
    img = img.to(device, dtype=dtype)
    latents = vae.encode(img).latent_dist.sample()
    
    shift = vae.config['shift_factor'] if vae.config['shift_factor'] is not None else 0
    latents = vae.config['scaling_factor'] * (latents - shift)
    
    num_channels_latents = pipe.transformer.config.in_channels // 4
    
    l_height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    l_width = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    packed_latents = pipe._pack_latents(latents, 1, num_channels_latents, l_height, l_width)
    
    packed_latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        packed_latents,
    )
    
    print("Calculating timestep weights...")
    
    torch.manual_seed(8675309)
    noise = torch.randn_like(packed_latents, device=device, dtype=dtype)
    
    # Create linear timesteps from 1000 to 0
    num_train_timesteps = 1000
    timesteps_torch = torch.linspace(1000, 1, num_train_timesteps, device='cpu')
    timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    
    timestep_weights = torch.zeros(num_train_timesteps, dtype=torch.float32, device=device)
    
    guidance = torch.full([1], 1.0, device=device, dtype=torch.float32)
    guidance = guidance.expand(latents.shape[0])
    
    pbar = tqdm(range(num_train_timesteps), desc="loss: 0.000000 scaler: 0.0000")
    for i in pbar:
        timestep = timesteps[i:i+1].to(device)
        t_01 = (timestep / 1000).to(device)
        t_01 = t_01.reshape(-1, 1, 1)
        noisy_latents = (1.0 - t_01) * packed_latents + t_01 * noise
        
        noise_pred = pipe.transformer(
            hidden_states=noisy_latents, # torch.Size([1, 4096, 64])
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        
        target = noise - packed_latents
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
        loss = loss
        
        # determine scaler to multiply loss by to make it 1
        scaler = 1.0 / (loss + 1e-6)
        
        timestep_weights[i] = scaler
        pbar.set_description(f"loss: {loss.item():.6f} scaler: {scaler.item():.4f}")
        
    print("normalizing timestep weights...")
    # normalize the timestep weights so they are a mean of 1.0
    timestep_weights = timestep_weights / timestep_weights.mean()
    timestep_weights = timestep_weights.cpu().numpy().tolist()
    
    print("Saving timestep weights...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timestep_weights, f, ensure_ascii=False)
        

print(f"Timestep weights saved to {output_path}")
print("Done!")
flush()
        
        
    
    
    
    
    
    
    
    
    
    