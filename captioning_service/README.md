# JoyCaption Service for AI-Toolkit

This service provides automatic image captioning using JoyCaption, a free and open-source Visual Language Model (VLM) for generating high-quality image descriptions.

## Features

- **Multiple Caption Styles**: Descriptive, casual, detailed, straightforward, Stable Diffusion prompts, and booru tags
- **Batch Processing**: Caption multiple images efficiently
- **Custom Prompts**: Use your own prompts for specific captioning needs
- **REST API**: Easy integration with the AI-Toolkit web interface
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 17GB VRAM (24GB+ recommended)
- **RAM**: 8GB+ system RAM
- **Storage**: ~10GB for model weights

### Software
- Python 3.8+
- CUDA-compatible PyTorch installation
- Git (for cloning JoyCaption)

## Installation

1. **Install Python dependencies**:
   ```bash
   cd captioning_service
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **The JoyCaption repository is already included** in the parent directory as a git submodule.

## Usage

### Starting the Service

**Quick Start** (recommended):
```bash
./captioning_service/start_service.sh
```

**Manual Start**:
```bash
cd captioning_service
source venv/bin/activate
python caption_server.py --host 127.0.0.1 --port 5000 --preload
```

### Configuration

Environment variables:
- `CAPTION_HOST`: Host to bind to (default: 127.0.0.1)
- `CAPTION_PORT`: Port to bind to (default: 5000)
- `CAPTION_MODEL`: Model to use (default: fancyfeast/llama-joycaption-beta-one-hf-llava)

### API Endpoints

#### Health Check
```bash
GET /health
```
Returns service status and GPU information.

#### Single Image Caption
```bash
POST /caption
Content-Type: application/json

{
  "image_path": "/path/to/image.jpg",
  "style": "descriptive",
  "max_new_tokens": 256,
  "temperature": 0.6
}
```

#### Batch Caption
```bash
POST /batch_caption
Content-Type: application/json

{
  "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
  "style": "descriptive",
  "max_new_tokens": 256,
  "temperature": 0.6
}
```

#### Available Prompts
```bash
GET /prompts
```
Returns available caption styles and their prompts.

## Caption Styles

- **descriptive**: Formal, detailed descriptions
- **casual**: Casual tone descriptions
- **detailed**: Long, comprehensive descriptions
- **straightforward**: Objective, concise captions
- **stable_diffusion**: SD-style prompts with tags
- **booru**: Danbooru-style tag lists

## Troubleshooting

### Service Won't Start
- Check GPU availability: `nvidia-smi`
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Check available VRAM (need 17GB+)

### Out of Memory Errors
- Reduce batch size
- Close other GPU applications
- Consider model quantization (4-bit/8-bit)

### Slow Performance
- Ensure GPU is being used (check `nvidia-smi`)
- Increase batch size if you have extra VRAM
- Use SSD storage for faster image loading

## Integration with AI-Toolkit

The captioning service integrates seamlessly with the AI-Toolkit web interface:

1. Start the captioning service
2. Navigate to any dataset in the web interface
3. Use the "Auto-Captioning" section to generate captions
4. Captions are automatically saved as .txt files alongside images

## Model Information

This service uses JoyCaption Beta One:
- **Model**: fancyfeast/llama-joycaption-beta-one-hf-llava
- **Base**: Llama 3.1 with vision capabilities
- **License**: Open source, unrestricted use
- **Performance**: Near GPT-4V quality for image captioning

For more information about JoyCaption, visit: https://github.com/fpgaminer/joycaption
