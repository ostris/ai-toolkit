#!/usr/bin/env python3
"""
AI captioning script for images and videos using Qwen2.5-VL.

Based on the captioning approach from:
https://github.com/seanhan19911990-source/ComfyUI-Seans-OmniTag

Usage:
    python scripts/caption_image.py --img_path /path/to/image.jpg \
        --trigger_word "ohwx person" \
        --system_prompt "Describe the image in detail."
"""

import argparse
import sys
import os
import json


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


def caption_image(img_path: str, trigger_word: str, system_prompt: str) -> str:
    """Generate a caption for an image or video using Qwen2.5-VL."""
    from PIL import Image
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    # Determine if this is a video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'}
    ext = os.path.splitext(img_path)[1].lower()
    is_video = ext in video_extensions

    # Load image or extract middle frame from video
    if is_video:
        image = get_video_middle_frame(img_path)
    else:
        image = Image.open(img_path).convert("RGB")

    # Build the prompt
    trigger = trigger_word.strip() if trigger_word and trigger_word.strip() else ""

    if not system_prompt or not system_prompt.strip():
        system_prompt = (
            "You are an expert image captioner. Describe the image in exhaustive detail. "
            "Be clinical and precise. Include all visible elements, their positions, colors, "
            "textures, and any relevant context."
        )

    user_message = system_prompt.strip()
    if trigger:
        user_message += f"\n\nIMPORTANT: Your caption MUST begin with the exact words: '{trigger}'"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_message},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Ensure caption starts with trigger word if provided
    if trigger and not caption.startswith(trigger):
        caption = f"{trigger}, {caption}"

    # Clean up
    model.to("cpu")
    del model
    del processor
    if device == "cuda":
        torch.cuda.empty_cache()

    return caption


def main():
    parser = argparse.ArgumentParser(description="Caption an image or video using Qwen VL model")
    parser.add_argument("--img_path", required=True, help="Path to the image or video file")
    parser.add_argument("--trigger_word", default="", help="Trigger word the caption should start with")
    parser.add_argument("--system_prompt", default="", help="System prompt with captioning instructions")
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(json.dumps({"error": f"File not found: {args.img_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        caption = caption_image(args.img_path, args.trigger_word, args.system_prompt)
        print(json.dumps({"caption": caption}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
