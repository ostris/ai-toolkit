#!/usr/bin/env python3
"""
Bulk AI captioning script for datasets using Qwen3-VL.

Reads captioning config and image list from stdin as JSON:
  { "images": [...], "trigger_word": "...", "system_prompt": "...", "model_id": "..." }

Outputs progress lines to stdout in the format: PROGRESS:captioned:total
Captions are saved as .txt files alongside the images.
"""

import sys
import json
import os
import gc

MODEL_LITE = "prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1"
MODEL_FULL = "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"

CORE_CAPTION_INSTRUCTION = (
    "You are a captioning tool. "
    "Write ONE single cohesive description of the overall scene and subject. "
    "Be objective, descriptive and precise."
)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'}


def get_txt_path(img_path: str) -> str:
    base, _ = os.path.splitext(img_path)
    return base + '.txt'


def get_video_middle_frame(video_path: str):
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
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def build_instruction(trigger: str, lora_focus: str) -> str:
    if lora_focus:
        if trigger:
            return (
                f"{CORE_CAPTION_INSTRUCTION}\n\n"
                f"Additional focus: {lora_focus}\n\n"
                f"Start the response with: {trigger}"
            )
        return f"{CORE_CAPTION_INSTRUCTION}\n\nAdditional focus: {lora_focus}"
    if trigger:
        return f"{CORE_CAPTION_INSTRUCTION}\n\nStart the response with: {trigger}"
    return CORE_CAPTION_INSTRUCTION


def generate_caption(model, processor, device, image, instruction: str, trigger: str) -> str:
    from qwen_vl_utils import process_vision_info
    import torch

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

    # Retry with greedy decoding if too short or lazy
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

    if not caption:
        caption = (
            f"{trigger}, a scene depicting visual content."
            if trigger
            else "A scene depicting visual content."
        )

    if trigger and not caption.startswith(trigger):
        caption = f"{trigger}, {caption}"

    return caption


def main():
    data = json.load(sys.stdin)
    image_paths = data.get('images', [])
    trigger_word = data.get('trigger_word', '').strip()
    system_prompt = data.get('system_prompt', '').strip()
    model_id = data.get('model_id', MODEL_LITE)

    if model_id not in (MODEL_LITE, MODEL_FULL):
        model_id = MODEL_LITE

    total = len(image_paths)
    if total == 0:
        print('PROGRESS:0:0', flush=True)
        return

    instruction = build_instruction(trigger_word, system_prompt)

    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from PIL import Image
    except ImportError as e:
        print(f'ERROR:Missing dependency: {e}', flush=True)
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model once for all images
    if device == 'cuda':
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=q_config,
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map='auto',
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    captioned = 0
    for img_path in image_paths:
        try:
            ext = os.path.splitext(img_path)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                image = get_video_middle_frame(img_path)
            else:
                image = Image.open(img_path).convert('RGB')

            caption = generate_caption(model, processor, device, image, instruction, trigger_word)

            txt_path = get_txt_path(img_path)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption)
        except Exception as e:
            print(f'WARN:Failed to caption {img_path}: {e}', flush=True)

        captioned += 1
        print(f'PROGRESS:{captioned}:{total}', flush=True)

    del model
    del processor
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
