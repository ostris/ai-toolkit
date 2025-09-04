from diffusers import DiffusionPipeline
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer 
from transformers import Qwen2_5_VLModel as TextEncoder
import json
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
import argparse
from pathlib import Path
import os
from typing import List
import time 



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen-Image image generation script.")
    parser.add_argument(
        "--model_path", # "Qwen/Qwen-Image"
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path", # /shareddata/dheyo/shivanvitha/ai-toolkit/output/aa_and_ab_qwen_image_1/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_1/snapshots/e3f588e48a4c67dff8dd173f5fb0343e86ddd405/tokenizer_0_aa_and_ab_qwen_image_1__000007200
        type=str,
        default=None,
        help="Path to updated tokenizer.",
    )
    parser.add_argument(
        "--text_encoder_path", # /shareddata/dheyo/shivanvitha/ai-toolkit/output/aa_and_ab_qwen_image_1/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_1/snapshots/e3f588e48a4c67dff8dd173f5fb0343e86ddd405/text_encoder_0_aa_and_ab_qwen_image_1__000007200
        type=str,
        default=None,
        help="Path to text encoder checkpoint.",
    )
    parser.add_argument(
        "--token_abstraction_json_path",
        type=str,
        # required=True,
        default=None,
        help="Path to token abstraction dict",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Path to transformer LoRA checkpoint.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation."
    )
    # parser.add_argument(
    #     "--height",
    #     type=int,
    #     default=1024,
    #     help="Output image height."
    # )
    # parser.add_argument(
    #     "--width",
    #     type=int,
    #     default=1024,
    #     help="Output image width."
    # )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Text prompt for generation."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, watermark",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="output.png",
        help="Path to save output image."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        help="Path to prompts.txt for bulk generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="""
            aspect_ratios = {
                "1:1": (1024, 1024),
                "16:9": (1664, 928),
                "9:16": (928, 1664),
                "4:3": (1472, 1140),
                "3:4": (1140, 1472),
                "3:2": (1584, 1056),
                "2:3": (1056, 1584),
            }
        """,
    )


    return parser.parse_args()


# lora_weights_path = "/shareddata/dheyo/shivanvitha/ai-toolkit/output/aa_and_ab_qwen_image_1/hub/models--shivmlops21--allu_arjun_and_alia_bhatt_1/snapshots/e3f588e48a4c67dff8dd173f5fb0343e86ddd405"

def convert_lora_weights_before_load(args:argparse.Namespace, state_dict, new_path):
    if args.token_abstraction_json_path:
        popped_key = state_dict.pop("emb_params")
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key.replace("diffusion_model.", "transformer.")
        new_sd[new_key] = value

    # dir_path = '/'.join(args.transformer_lora_path.split('/')[:-1]) 
    # cleaned_safetensors_file = args.transformer_lora_path.split('/')[-1].replace(".safetensors", "_cleaned.safetensors")
    save_file(new_sd, f"{new_path}")
    return new_sd



def load_pipeline (args:argparse.Namespace):
    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(device)


    dir_path = '/'.join(args.transformer_lora_path.split('/')[:-1]) 
    cleaned_safetensors_file = args.transformer_lora_path.split('/')[-1].replace(".safetensors", "_cleaned.safetensors")
    new_path = f"{dir_path}/{cleaned_safetensors_file}"
    print(f"{new_path} exists")

    if not os.path.exists(new_path):
        print(f"Creating {new_path}...")
        state_dict = load_file(f"{args.transformer_lora_path}")
        state_dict = convert_lora_weights_before_load(args, state_dict, new_path)

    pipe.load_lora_weights(dir_path, weight_name=cleaned_safetensors_file)


    # loading new tokenizer and text encoder here!!!!
    if args.tokenizer_path and args.text_encoder_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

        text_encoder = TextEncoder.from_pretrained(args.text_encoder_path,
                                                ignore_mismatched_sizes=True,
                                                torch_dtype=torch_dtype).to(device)

        text_encoder.resize_token_embeddings(len(tokenizer))
        pipe.tokenizer = tokenizer

        pipe.text_encoder = text_encoder 

    return pipe

def main (args:argparse.Namespace, prompts: List) -> None:
    pipe = load_pipeline(args)
    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition." # for english prompt
    }

    if args.token_abstraction_json_path:
        with open(args.token_abstraction_json_path, "r") as file:
            representation_tokens = json.load(file)

        special_tokens = list(representation_tokens.keys())

    aspect_ratios = {
        "1:1": (1024, 1024),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios[args.aspect_ratio]

    for idx, prompt in enumerate(prompts):
        if args.token_abstraction_json_path:
            for special_token in special_tokens:
                prompt = prompt.replace(special_token, representation_tokens[special_token][0].replace(" ", ''))

        print(prompt)

        images = pipe(
            num_images_per_prompt=args.num_images_per_prompt,
            prompt=prompt + positive_magic["en"],
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed)
        ).images

        os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
        timestamp = str(time.strftime("%d-%m-%y_%H-%M-%S"))

        for image_id, image in enumerate(images):
            file_path = f"{args.output_image_path.replace('.png', '')}_{idx}_{image_id}_{timestamp}.png"
            image.save(file_path)
            print(f"Saved {file_path}")


if __name__ == '__main__':
    args = parse_args()
    if not args.instruction and not args.prompts_path:
        raise ValueError("Either --instruction or --prompts_path has to be specified, both are None")

    if args.prompts_path:
        prompts_path = Path(args.prompts_path)
        prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    else:
        prompts = [args.instruction]
    main(args, prompts)
