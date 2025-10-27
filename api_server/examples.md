# Examples

This walkthrough shows how to start a training session with the new step budget API, let it run for 100 steps, resume with another 100-step allocation, and finally abort the job

## 1. Create the session

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "sdxl-lora-demo",
    "maxSteps": 3000,
    "config": {
      "job": "extension",
      "config": {
        "name": "sdxl-lora-demo",
        "process": [
          {
            "type": "diffusion_trainer",
            "training_folder": "/work/output",
            "device": "cuda",
            "trigger_word": null,
            "performance_log_every": 10,
            "network": {
              "type": "lora",
              "linear": 32,
              "linear_alpha": 32,
              "conv": 16,
              "conv_alpha": 16,
              "lokr_full_rank": true,
              "lokr_factor": 16,
              "network_kwargs": {
                "ignore_if_contains": []
              }
            },
            "save": {
              "dtype": "bf16",
              "save_every": 100,
              "max_step_saves_to_keep": 300,
              "save_format": "diffusers",
              "push_to_hub": false
            },
            "datasets": [
              {
                "folder_path": "/work/pop_tart",
                "mask_path": null,
                "mask_min_value": 0.1,
                "default_caption": "",
                "caption_ext": "txt",
                "caption_dropout_rate": 0.1,
                "shuffle_tokens": true,
                "token_dropout_rate": 0.01,
                "is_reg": false,
                "network_weight": 1,
                "bucket_step_size": 8,
                "bucket_no_upscale": true,
                "random_crop": false,
                "center_crop": false,
                "resolution": [1024],
                "controls": [],
                "cache_latents_to_disk": false,
                "flip_x": true
              }
            ],
            "train": {
              "batch_size": 2,
              "bypass_guidance_embedding": false,
              "steps": 3000,
              "gradient_accumulation": 1,
              "train_unet": true,
              "train_text_encoder": true,
              "gradient_checkpointing": true,
              "noise_scheduler": "ddpm",
              "optimizer": "adamw8bit",
              "timestep_type": "sigmoid",
              "content_or_style": "balanced",
              "optimizer_params": {
                "weight_decay": 0.01,
                "betas": [0.9, 0.999]
              },
              "unload_text_encoder": false,
              "lr": 0.0001,
              "text_encoder_lr": 0.00005,
              "lr_scheduler": "cosine",
              "lr_scheduler_kwargs": {
                "eta_min": 0.00001
              },
              "ema_config": {
                "use_ema": true,
                "ema_decay": 0.99
              },
              "dtype": "bf16",
              "mixed_precision": "bf16",
              "diff_output_preservation": false,
              "diff_output_preservation_multiplier": 1,
              "diff_output_preservation_class": "person",
              "max_grad_norm": 1,
              "noise_offset": 0.1,
              "min_snr_gamma": 5,
              "zero_terminal_snr": true,
              "scale_weight_norms": 1,
              "cache_text_encoder_outputs": true,
              "cache_text_encoder_outputs_to_disk": false
            },
            "model": {
              "name_or_path": "/work/models/hassakuXLIllustrious_v22.safetensors",
              "quantize": false,
              "quantize_te": false,
              "arch": "sdxl",
              "low_vram": false,
              "model_kwargs": {}
            },
            "sample": {
              "sampler": "ddpm",
              "sample_every": 100,
              "width": 1024,
              "height": 1024,
              "neg": "3d, ugly, blurry, low quality, horror",
              "seed": 43,
              "walk_seed": true,
              "guidance_scale": 6,
              "sample_steps": 35,
              "num_frames": 1,
              "fps": 1,
              "cfg_rescale": 0,
              "samples": [
                { "prompt": "TOK," }
              ]
            },
            "meta": {
              "name": "[name]",
              "version": "1.0"
            }
          }
        ]
      }
    }
  }'
```

## 2. Authorise the first 100 steps

```bash
curl -X POST http://localhost:8000/sessions/sdxl-lora-demo/steps \
  -H "Content-Type: application/json" \
  -d '{ "steps": 100 }'
```

## 3. Resume with another 100 steps

```bash
curl -X POST http://localhost:8000/sessions/sdxl-lora-demo/steps \
  -H "Content-Type: application/json" \
  -d '{ "steps": 100 }'
```

## 4. Stream logs (optional and may delete it as not much useful)

```bash
curl -s -N http://localhost:8000/sessions/sdxl-lora-demo/logs/stream
```

## 5. Abort the session

```bash
curl -X POST http://localhost:8000/sessions/sdxl-lora-demo/abort
```

After aborting, the trainer saves its latest checkpoint and releases GPU resources
## 6. Resume on a different GPU

1. Ask the server for the latest status to discover the checkpoint path (look for `last_checkpoint_path`):

```bash
curl http://localhost:8000/sessions/sdxl-lora-demo
```

2. Abort and delete the in-memory session once the GPU is free (clears API bookkeeping but leaves checkpoints on disk):

```bash
curl -X POST http://localhost:8000/sessions/sdxl-lora-demo/abort
curl -X DELETE http://localhost:8000/sessions/sdxl-lora-demo
```

3. On the a new host/instance recreate the session using the same training payload. As long as `save.training_folder` still contains the saved weights and the `job` name is unchanged, the trainer automatically reloads the latest `.safetensors` checkpoint when it starts

4. Allocate another step budget just like before:

```bash
curl -X POST http://localhost:8000/sessions/sdxl-lora-demo/steps \
  -H "Content-Type: application/json" \
  -d '{ "steps": 100 }'
```

---

# Image and Video Captioning API

The API server includes endpoints for generating captions using **Florence2** or **JoyCaption** models. Both models support images and videos.

## Florence2 Examples

### 1. Caption an image with Florence2 (default model)

```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/workspaces/ai-toolkit/fennec_girl_flowers.png"
  }'
```

### 3. Caption a video with Florence2

```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/workspaces/ai-toolkit/video.mp4",
    "model_type": "florence2",
    "num_frames": 8,
    "sample_method": "uniform",
    "combine_method": "first",
    "max_new_tokens": 1024,
    "num_beams": 3,
    "task": "<DETAILED_CAPTION>"
  }'
```

## JoyCaption Examples

### 5. Caption an image with JoyCaption

```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/workspaces/ai-toolkit/fennec_girl_flowers.png",
    "model_type": "joycaption"
  }'
```

### 6. JoyCaption with custom prompt

```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "model_type": "joycaption",
    "prompt": "Write a long descriptive caption for this image in a formal tone.",
    "max_new_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.9
  }'
```

### 9. Caption a video with JoyCaption

```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/workspaces/ai-toolkit/video.mp4",
    "model_type": "joycaption",
    "prompt": "Describe what is happening in this video in a narrative style.",
    "num_frames": 8,
    "sample_method": "uniform",
    "combine_method": "longest",
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": 512
  }'
```

## Model Management

### 13. Unload caption model to free GPU memory

```bash
curl -X POST http://localhost:8000/caption/unload \
  -H "Content-Type: application/json"
```

## Complete Parameter Reference

### Common Parameters (Both Models)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | string | null | Path to local image file |
| `image_url` | string | null | URL to image file |
| `video_path` | string | null | Path to local video file |
| `video_url` | string | null | URL to video file |
| `model_type` | string | "florence2" | Model to use: "florence2" or "joycaption" |
| `model_path` | string | varies | Custom model path (overrides default) |
| `max_new_tokens` | integer | 1024 (florence2)<br>512 (joycaption) | Maximum tokens to generate |
| `num_frames` | integer | 8 | Number of frames to extract from video |
| `sample_method` | string | "uniform" | Frame sampling: "uniform" or "first" |
| `combine_method` | string | "first" | Caption combining: "first", "longest", or "combined" |

### Florence2-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_beams` | integer | 3 | Number of beams for beam search |
| `task` | string | "&lt;DETAILED_CAPTION&gt;" | Task type: "&lt;CAPTION&gt;", "&lt;DETAILED_CAPTION&gt;", "&lt;MORE_DETAILED_CAPTION&gt;" |

### JoyCaption-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "Write a long descriptive caption..." | Custom prompt for caption generation |
| `temperature` | float | 0.6 | Sampling temperature (0.0-2.0)<br>Lower = more focused, Higher = more creative |
| `top_p` | float | 0.9 | Nucleus sampling parameter (0.0-1.0)<br>Lower = more focused, Higher = more diverse |
