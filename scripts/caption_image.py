#!/usr/bin/env python3
"""
AI captioning script for images and videos using Qwen3-VL.

Based on the captioning approach from:
https://github.com/seanhan19911990-source/ComfyUI-Seans-OmniTag

Usage:
    python scripts/caption_image.py --img_path /path/to/image.jpg \
        --trigger_word "ohwx person" \
        --system_prompt "Describe the image in detail." \
        --model_id "prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1"
"""

import argparse
import sys
import os
import json
import gc

# Core captioning behaviour — controls HOW the model processes frames.
# The user's system_prompt (lora_focus) controls WHAT to describe.
CORE_CAPTION_INSTRUCTION = (
    "You are a captioning tool. "
    "Write ONE single cohesive description of the overall scene and subject. "
    "Be objective, descriptive and precise."
)

MODEL_LITE = "prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1"
MODEL_FULL = "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"


def get_video_middle_frame(video_path: str):
    """Extract the middle frame from a video file."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read frame from video")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def caption_image(img_path: str, trigger_word: str, system_prompt: str, model_id: str) -> str:
    """Generate a caption for an image or video using Qwen3-VL."""
    from PIL import Image
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    # Determine if this is a video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'}
    ext = os.path.splitext(img_path)[1].lower()
    is_video = ext in video_extensions

    # Load image or extract middle frame from video
    if is_video:
        image = get_video_middle_frame(img_path)
    else:
        image = Image.open(img_path).convert("RGB")

    trigger = trigger_word.strip() if trigger_word and trigger_word.strip() else ""

    lora_focus = system_prompt.strip() if system_prompt and system_prompt.strip() else ""

    # Build the instruction matching the OmniTag approach
    if lora_focus:
        instruction = (
            f"{CORE_CAPTION_INSTRUCTION}\n\n"
            f"Additional focus: {lora_focus}\n\n"
            f"Start the response with: {trigger}" if trigger else
            f"{CORE_CAPTION_INSTRUCTION}\n\n"
            f"Additional focus: {lora_focus}"
        )
    else:
        instruction = (
            f"{CORE_CAPTION_INSTRUCTION}\n\nStart the response with: {trigger}"
            if trigger else CORE_CAPTION_INSTRUCTION
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use 4-bit quantization on CUDA (matches OmniTag approach)
    if device == "cuda":
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=q_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_in],
        images=img_in,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.12,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    caption = processor.batch_decode(
        [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)],
        skip_special_tokens=True,
    )[0].strip()

    # Retry with greedy decoding if caption is too short or lazy
    if not caption or len(caption) < 30:
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.25,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        caption = processor.batch_decode(
            [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)],
            skip_special_tokens=True,
        )[0].strip()

    # Fallback if still empty
    if not caption:
        caption = f"{trigger}, a scene depicting visual content." if trigger else "A scene depicting visual content."

    # Ensure caption starts with trigger word if provided and model didn't include it
    if trigger and not caption.startswith(trigger):
        caption = f"{trigger}, {caption}"

    # Clean up
    del model
    del processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return caption


def main():
    parser = argparse.ArgumentParser(description="Caption an image or video using Qwen3-VL model")
    parser.add_argument("--img_path", required=True, help="Path to the image or video file")
    parser.add_argument("--trigger_word", default="", help="Trigger word the caption should start with")
    parser.add_argument("--system_prompt", default="", help="System prompt with captioning instructions")
    parser.add_argument(
        "--model_id",
        default=MODEL_LITE,
        choices=[MODEL_LITE, MODEL_FULL],
        help="Qwen3-VL model to use for captioning",
    )
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(json.dumps({"error": f"File not found: {args.img_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        caption = caption_image(args.img_path, args.trigger_word, args.system_prompt, args.model_id)
        print(json.dumps({"caption": caption}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
