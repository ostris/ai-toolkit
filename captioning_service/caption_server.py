#!/usr/bin/env python3
"""
JoyCaption captioning service for AI-Toolkit
Provides a REST API for generating image captions using JoyCaption
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import threading
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, LlavaForConditionalGeneration
from tqdm import tqdm

# Add the joycaption directory to the path so we can import from it
sys.path.append(str(Path(__file__).parent.parent / "joycaption"))

app = Flask(__name__)

# Global variables for the model
tokenizer = None
llava_model = None
model_loaded = False
loading_progress = {
    'status': 'idle',  # idle, downloading, loading, ready, error
    'message': '',
    'progress': 0.0,
    'current_file': '',
    'files_downloaded': 0,
    'total_files': 0
}

# Default prompts for different caption styles
DEFAULT_PROMPTS = {
    "descriptive": "Write a descriptive caption for this image in a formal tone.",
    "casual": "Write a descriptive caption for this image in a casual tone.",
    "detailed": "Write a long detailed description for this image.",
    "straightforward": "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
    "stable_diffusion": "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    "booru": "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text."
}

# JoyCaption prompt templates
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."

def build_prompt(caption_type, caption_length, extra_options, name_input):
    """Build the prompt based on JoyCaption's prompt building logic"""

    # Choose the right template row in CAPTION_TYPE_MAP
    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1  # numeric-word-count template
    else:
        map_idx = 2  # length descriptor template

    # Get the base prompt
    templates = CAPTION_TYPE_MAP.get(caption_type, CAPTION_TYPE_MAP["Descriptive"])
    prompt = templates[map_idx]

    # Add extra options
    if extra_options:
        prompt += " " + " ".join(extra_options)

    # Format the prompt
    return prompt.format(
        name=name_input or "{NAME}",
        length=caption_length,
        word_count=caption_length,
    )

class ProgressCapture:
    """Capture tqdm progress from transformers downloads"""
    def __init__(self):
        self.current_file = ""
        self.files_downloaded = 0
        self.total_files = 0
        self.overall_progress = 0.0

    def update_from_tqdm_output(self, line):
        """Parse tqdm output to extract progress"""
        # Look for patterns like "Fetching 4 files: 25%|██▌       | 1/4 [00:30<01:30, 30.0s/it]"
        if "Fetching" in line and "files:" in line:
            # Extract total files
            match = re.search(r'Fetching (\d+) files:', line)
            if match:
                self.total_files = int(match.group(1))

        # Look for progress percentage
        progress_match = re.search(r'(\d+)%\|[^|]*\|\s*(\d+)/(\d+)', line)
        if progress_match:
            percent = int(progress_match.group(1))
            current = int(progress_match.group(2))
            total = int(progress_match.group(3))

            self.files_downloaded = current
            self.total_files = total
            self.overall_progress = percent / 100.0

            if current < total:
                self.current_file = f"Downloading file {current}/{total}"
            else:
                self.current_file = "Download complete"

progress_capture = ProgressCapture()

def load_model(model_name: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"):
    """Load the JoyCaption model"""
    global tokenizer, llava_model, model_loaded, loading_progress, progress_capture

    if model_loaded:
        loading_progress['status'] = 'ready'
        return True

    try:
        loading_progress['status'] = 'downloading'
        loading_progress['message'] = 'Initializing model download...'
        loading_progress['progress'] = 0.0

        logging.info(f"Loading JoyCaption model: {model_name}")

        # Load tokenizer first (usually quick)
        loading_progress['message'] = 'Loading tokenizer...'
        loading_progress['progress'] = 0.05
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Load model - this is where the heavy downloading/loading happens
        loading_progress['status'] = 'loading'
        loading_progress['message'] = 'Loading model (this may take several minutes)...'
        loading_progress['progress'] = 0.1

        # Start a thread to monitor actual progress
        def update_loading_progress():
            start_time = time.time()
            last_progress_update = start_time

            while not model_loaded:
                elapsed = time.time() - start_time

                # Phase 1: Initial setup (first 5 seconds)
                if elapsed < 5:
                    loading_progress['progress'] = 0.05
                    loading_progress['message'] = 'Initializing model download...'

                # Phase 2: Download phase (5 seconds to 4 minutes)
                elif elapsed < 240:  # 4 minutes for download
                    # Progress from 5% to 80% over 4 minutes
                    download_progress = 0.05 + ((elapsed - 5) / 235) * 0.75
                    loading_progress['progress'] = min(0.80, download_progress)

                    if elapsed < 30:
                        loading_progress['message'] = 'Downloading model configuration and tokenizer...'
                    elif elapsed < 120:
                        loading_progress['message'] = 'Downloading model weights (this may take several minutes)...'
                    elif elapsed < 200:
                        loading_progress['message'] = 'Downloading remaining model files...'
                    else:
                        loading_progress['message'] = 'Download nearly complete...'

                # Phase 3: Loading into memory (after 4 minutes)
                else:
                    if loading_progress['progress'] < 0.85:
                        loading_progress['progress'] = 0.85
                        loading_progress['message'] = 'Loading model into GPU memory...'
                    elif elapsed > 270:  # After 4.5 minutes
                        loading_progress['progress'] = 0.95
                        loading_progress['message'] = 'Initializing model layers... Almost ready!'

                time.sleep(2)  # Update every 2 seconds
                if model_loaded:
                    break

        # Start progress updates in background
        progress_thread = threading.Thread(target=update_loading_progress, daemon=True)
        progress_thread.start()

        # The actual model loading - this will download ~10GB+ of files if not cached
        # Force to use GPU 1 to avoid conflicts with training jobs on GPU 0
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 1}  # Force to GPU 1
        )

        loading_progress['message'] = 'Finalizing model setup...'
        loading_progress['progress'] = 0.98
        llava_model.eval()

        model_loaded = True
        loading_progress['status'] = 'ready'
        loading_progress['message'] = 'Model ready for captioning'
        loading_progress['progress'] = 1.0
        logging.info("Model loaded successfully")
        return True

    except Exception as e:
        loading_progress['status'] = 'error'
        loading_progress['message'] = f'Failed to load model: {str(e)}'
        logging.error(f"Failed to load model: {e}")
        return False

def preprocess_image(image_path: str) -> Optional[torch.Tensor]:
    """Preprocess an image for JoyCaption"""
    try:
        image = Image.open(image_path)
        if image.size != (384, 384):
            image = image.resize((384, 384), Image.LANCZOS)
        image = image.convert("RGB")
        pixel_values = TVF.pil_to_tensor(image)
        return pixel_values
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        return None

def generate_caption(image_path: str, prompt: str, **kwargs) -> Optional[str]:
    """Generate a caption for a single image"""
    global tokenizer, llava_model
    
    if not model_loaded:
        return None
    
    # Default generation parameters
    max_new_tokens = kwargs.get('max_new_tokens', 256)
    temperature = kwargs.get('temperature', 0.6)
    top_p = kwargs.get('top_p', 0.9)
    top_k = kwargs.get('top_k', None)
    do_sample = kwargs.get('do_sample', True)
    
    try:
        # Preprocess image
        pixel_values = preprocess_image(image_path)
        if pixel_values is None:
            return None
        
        # Build conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        # Format conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        # Tokenize the conversation
        convo_tokens = tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

        # Repeat the image tokens (crucial for JoyCaption)
        image_token_id = llava_model.config.image_token_index
        image_seq_length = llava_model.config.image_seq_length

        input_tokens = []
        for token in convo_tokens:
            if token == image_token_id:
                input_tokens.extend([image_token_id] * image_seq_length)
            else:
                input_tokens.append(token)

        # Create tensors
        input_ids = torch.tensor([input_tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Move to appropriate devices
        vision_device = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
        language_device = llava_model.language_model.get_input_embeddings().weight.device

        input_ids = input_ids.to(language_device)
        attention_mask = attention_mask.to(language_device)
        
        # Prepare pixel values
        pixel_values = pixel_values.unsqueeze(0).to(vision_device)
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype)
        
        # Generate caption
        with torch.no_grad():
            generate_ids = llava_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=True,
                suppress_tokens=None,
            )[0]
        
        # Trim off the prompt
        generate_ids = generate_ids[input_ids.shape[1]:]
        
        # Decode caption
        caption = tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return caption.strip()
        
    except Exception as e:
        logging.error(f"Failed to generate caption for {image_path}: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'loading_progress': loading_progress
    })

@app.route('/caption', methods=['POST'])
def caption_image():
    """Caption a single image"""
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({'error': 'image_path is required'}), 400
        
        image_path = data['image_path']
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file not found: {image_path}'}), 404
        
        # Get prompt - support both old style and new JoyCaption parameters
        custom_prompt = data.get('prompt')

        if custom_prompt:
            prompt = custom_prompt
        else:
            # Check for JoyCaption parameters
            caption_type = data.get('caption_type')
            caption_length = data.get('caption_length', 'long')
            extra_options = data.get('extra_options', [])
            name_input = data.get('name_input', '')

            if caption_type and caption_type in CAPTION_TYPE_MAP:
                # Use JoyCaption prompt building
                prompt = build_prompt(caption_type, caption_length, extra_options, name_input)
            else:
                # Fall back to old style
                prompt_style = data.get('style', 'descriptive')
                if prompt_style in DEFAULT_PROMPTS:
                    prompt = DEFAULT_PROMPTS[prompt_style]
                else:
                    return jsonify({'error': f'Unknown prompt style: {prompt_style}'}), 400
        
        # Generation parameters
        gen_params = {
            'max_new_tokens': data.get('max_new_tokens', 256),
            'temperature': data.get('temperature', 0.6),
            'top_p': data.get('top_p', 0.9),
            'top_k': data.get('top_k'),
            'do_sample': data.get('do_sample', True)
        }
        
        # Generate caption
        start_time = time.time()
        caption = generate_caption(image_path, prompt, **gen_params)
        generation_time = time.time() - start_time
        
        if caption is None:
            return jsonify({'error': 'Failed to generate caption'}), 500
        
        return jsonify({
            'success': True,
            'caption': caption,
            'prompt_used': prompt,
            'generation_time': generation_time,
            'image_path': image_path
        })
        
    except Exception as e:
        logging.error(f"Error in caption_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_caption', methods=['POST'])
def batch_caption():
    """Caption multiple images"""
    try:
        data = request.get_json()
        
        if not data or 'image_paths' not in data:
            return jsonify({'error': 'image_paths is required'}), 400
        
        image_paths = data['image_paths']
        if not isinstance(image_paths, list):
            return jsonify({'error': 'image_paths must be a list'}), 400
        
        # Get prompt - support both old style and new JoyCaption parameters
        custom_prompt = data.get('prompt')

        if custom_prompt:
            prompt = custom_prompt
        else:
            # Check for JoyCaption parameters
            caption_type = data.get('caption_type')
            caption_length = data.get('caption_length', 'long')
            extra_options = data.get('extra_options', [])
            name_input = data.get('name_input', '')

            if caption_type and caption_type in CAPTION_TYPE_MAP:
                # Use JoyCaption prompt building
                prompt = build_prompt(caption_type, caption_length, extra_options, name_input)
            else:
                # Fall back to old style
                prompt_style = data.get('style', 'descriptive')
                if prompt_style in DEFAULT_PROMPTS:
                    prompt = DEFAULT_PROMPTS[prompt_style]
                else:
                    return jsonify({'error': f'Unknown prompt style: {prompt_style}'}), 400
        
        # Generation parameters
        gen_params = {
            'max_new_tokens': data.get('max_new_tokens', 256),
            'temperature': data.get('temperature', 0.6),
            'top_p': data.get('top_p', 0.9),
            'top_k': data.get('top_k'),
            'do_sample': data.get('do_sample', True)
        }
        
        results = []
        total_time = 0
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': 'File not found'
                })
                continue
            
            start_time = time.time()
            caption = generate_caption(image_path, prompt, **gen_params)
            generation_time = time.time() - start_time
            total_time += generation_time
            
            if caption is None:
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': 'Failed to generate caption'
                })
            else:
                results.append({
                    'image_path': image_path,
                    'success': True,
                    'caption': caption,
                    'generation_time': generation_time
                })
        
        successful_captions = sum(1 for r in results if r['success'])
        
        return jsonify({
            'success': True,
            'results': results,
            'total_images': len(image_paths),
            'successful_captions': successful_captions,
            'failed_captions': len(image_paths) - successful_captions,
            'total_time': total_time,
            'prompt_used': prompt
        })
        
    except Exception as e:
        logging.error(f"Error in batch_caption: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress', methods=['GET'])
def get_loading_progress():
    """Get detailed loading progress"""
    return jsonify(loading_progress)

@app.route('/prompts', methods=['GET'])
def get_available_prompts():
    """Get available prompt styles"""
    return jsonify({
        'prompts': DEFAULT_PROMPTS,
        'styles': list(DEFAULT_PROMPTS.keys()),
        'joycaption_types': list(CAPTION_TYPE_MAP.keys())
    })

if __name__ == '__main__':
    import threading

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description='JoyCaption API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--model', default='fancyfeast/llama-joycaption-beta-one-hf-llava', help='Model to use')
    parser.add_argument('--preload', action='store_true', help='Preload the model on startup')

    args = parser.parse_args()

    # Start model loading in background if preload is requested
    if args.preload:
        def load_model_background():
            if not load_model(args.model):
                logging.error("Failed to preload model")
                loading_progress['status'] = 'error'
                loading_progress['message'] = 'Failed to load model'

        # Start loading in background thread
        loading_thread = threading.Thread(target=load_model_background, daemon=True)
        loading_thread.start()

    # Start the server immediately
    logging.info(f"Starting JoyCaption API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
