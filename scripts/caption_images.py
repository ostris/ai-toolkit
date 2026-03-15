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

MODEL_LITE = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_FULL = "Qwen/Qwen3-VL-8B-Instruct"

ALLOWED_MODELS = {
    MODEL_LITE,
    MODEL_FULL,
    "prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1",
    "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it",
}


CORE_CAPTION_INSTRUCTION = (
    "You are a captioning tool. "
    "Write ONE single cohesive description of the overall scene and subject. "
    "Be objective, descriptive and precise."
)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'}

QUORUM_SYNTHESIS_PROMPT = (
    "Below are 5 candidate captions for this image. Generate a final caption "
    "using ONLY details that appear in at least 3 of the 5 candidates. "
    "Discard any detail that appears in fewer than 3. Write a single cohesive "
    "caption — do not number items or mention the candidates."
)

QUORUM_TEMPERATURES = [0.6, 0.7, 0.8, 0.9, 1.0]


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


def _generate_single(model, processor, device, image, text_instruction, temperature=0.7, greedy=False):
    """Run a single generation pass and return the decoded caption."""
    from qwen_vl_utils import process_vision_info
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_instruction},
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

    gen_kwargs = dict(
        max_new_tokens=1024,
        repetition_penalty=1.12,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    if greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, **gen_kwargs)

    return processor.batch_decode(
        [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)],
        skip_special_tokens=True,
    )[0].strip()


def generate_caption(model, processor, device, image, instruction: str, trigger: str, quorum: bool = False) -> str:
    if quorum:
        # Generate 5 candidate captions at varying temperatures
        candidates = []
        for temp in QUORUM_TEMPERATURES:
            c = _generate_single(model, processor, device, image, instruction, temperature=temp)
            if c:
                candidates.append(c)

        if len(candidates) >= 2:
            numbered = "\n".join(f"Candidate {i+1}: {c}" for i, c in enumerate(candidates))
            synthesis_instruction = f"{QUORUM_SYNTHESIS_PROMPT}\n\n{numbered}"
            if trigger:
                synthesis_instruction += f"\n\nStart the response with: {trigger}"
            caption = _generate_single(model, processor, device, image, synthesis_instruction, greedy=True)
        else:
            caption = candidates[0] if candidates else ""
    else:
        caption = _generate_single(model, processor, device, image, instruction)

        # Retry with greedy decoding if too short or lazy
        if not caption or len(caption) < 30:
            caption = _generate_single(model, processor, device, image, instruction, greedy=True)

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
    quorum = data.get('quorum', False)

    if model_id not in ALLOWED_MODELS:
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

    # Check if model is already cached
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_files_only=True)
    except Exception:
        print('STATUS:downloading', flush=True)

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

            caption = generate_caption(model, processor, device, image, instruction, trigger_word, quorum=quorum)

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
