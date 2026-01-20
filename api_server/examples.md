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

### Caption an image with Florence2 (default model)

```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/workspaces/ai-toolkit/fennec_girl_flowers.png"
  }'
```

### Caption a video with Florence2

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

### Caption a video with Florence2 + audio

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
    "task": "<DETAILED_CAPTION>",
    "do_audio": true
  }'
```

## JoyCaption Examples

### Caption an image with JoyCaption

```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/workspaces/ai-toolkit/fennec_girl_flowers.png",
    "model_type": "joycaption"
  }'
```

### JoyCaption with custom prompt

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

### Caption a video with JoyCaption

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

### Caption a video with JoyCaption + audio

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
    "max_new_tokens": 512,
    "do_audio": true,
    "audio_prompt": "Provide a concise but detailed caption of the audio. Describe any speech (content summary if clear), speaker count, perceived gender/age, tone/emotion, language/accent, and notable non-speech sounds such as music (genre/instruments/tempo/mood) or environmental noises. If there is no speech, focus on the audio events and atmosphere."
  }'
```

## Model Management

### Unload caption model to free GPU memory

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
| `do_audio` | boolean | false | Also caption audio when using video inputs |

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

### Video Audio Captioning Parameters (Video Only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_model_path` | string | "mispeech/midashenglm-7b-0804-fp8" | Custom audio model path |
| `audio_prompt` | string | "Provide a concise but detailed caption of the audio. Describe any speech (content summary if clear), speaker count, perceived gender/age, tone/emotion, language/accent, and notable non-speech sounds such as music (genre/instruments/tempo/mood) or environmental noises. If there is no speech, focus on the audio events and atmosphere." | Prompt for audio captioning |
| `audio_max_new_tokens` | integer | 256 | Maximum tokens for audio captioning |
| `audio_temperature` | float | 0.2 | Sampling temperature for audio captioning |
| `audio_top_p` | float | 0.9 | Top-p for audio captioning |
| `audio_num_beams` | integer | 1 | Beam count for audio captioning |
| `audio_do_sample` | boolean | true | Whether to sample for audio captioning |
| `audio_repetition_penalty` | float | null | Repetition penalty for audio captioning |
| `audio_target_sample_rate` | integer | 16000 | Sample rate when extracting audio from video |
| `audio_max_audio_seconds` | float | null | Max duration for extracted audio |

---

# Audio Captioning API

The API server includes an endpoint for generating captions from audio or video (audio extracted).

## Audio Captioning Examples

### Caption an audio file

```bash
curl -X POST http://localhost:8000/audio/caption \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/workspaces/ai-toolkit/audio.wav"
  }'
```

### Caption a video (audio extracted)

```bash
curl -X POST http://localhost:8000/audio/caption \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/workspaces/ai-toolkit/video.mp4",
    "target_sample_rate": 16000,
    "max_audio_seconds": 30
  }'
```

### Caption an audio URL with custom options

```bash
curl -X POST http://localhost:8000/audio/caption \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/sample.wav",
    "prompt": "Provide a concise but detailed caption of the audio. Describe any speech (content summary if clear), speaker count, perceived gender/age, tone/emotion, language/accent, and notable non-speech sounds such as music (genre/instruments/tempo/mood) or environmental noises. If there is no speech, focus on the audio events and atmosphere.",
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9,
    "num_beams": 1,
    "do_sample": true
  }'
```

## Audio Model Management

### Unload audio caption model

```bash
curl -X POST http://localhost:8000/audio/unload \
  -H "Content-Type: application/json"
```

### Free all VRAM (sessions + caption + tag + audio)

```bash
curl -X POST http://localhost:8000/vram/free \
  -H "Content-Type: application/json"
```

## Audio Captioning Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_path` | string | null | Path to local audio file |
| `audio_url` | string | null | URL to audio file |
| `video_path` | string | null | Path to local video file (audio extracted) |
| `video_url` | string | null | URL to video file (audio extracted) |
| `model_path` | string | "mispeech/midashenglm-7b-0804-fp8" | Custom model path (overrides default) |
| `prompt` | string | "Provide a concise but detailed caption of the audio. Describe any speech (content summary if clear), speaker count, perceived gender/age, tone/emotion, language/accent, and notable non-speech sounds such as music (genre/instruments/tempo/mood) or environmental noises. If there is no speech, focus on the audio events and atmosphere." | Prompt for captioning |
| `max_new_tokens` | integer | 256 | Maximum tokens to generate |
| `temperature` | float | 0.2 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling parameter |
| `num_beams` | integer | 1 | Number of beams for generation |
| `do_sample` | boolean | true | Whether to sample during generation |
| `repetition_penalty` | float | null | Repetition penalty |
| `target_sample_rate` | integer | 16000 | Sample rate for extracted audio |
| `max_audio_seconds` | float | null | Max audio duration to process in seconds |

Notes:

- Provide either `audio_*` or `video_*`, not both.
- Video inputs are decoded to audio before captioning.

---

# Tagging API

The API server also supports WD14 tagger inference. The default model is `wd14-vit.v1` (SmilingWolf/wd-v1-4-vit-tagger).

## Tagging Examples

### Tag an image

```bash
curl -X POST http://localhost:8000/tag \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["/workspaces/ai-toolkit/cat.png"]
  }'
```

### Tag an image from URL

```bash
curl -X POST http://localhost:8000/tag \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {
        "media_url": "https://example.com/cat.png"
      }
    ]
  }'
```

### Tag using a local model directory

```bash
curl -X POST http://localhost:8000/tag \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/wd14-vit-v1",
    "input": ["/workspaces/ai-toolkit/cat.png"]
  }'
```

### Tag a video with custom thresholds

```bash
curl -X POST http://localhost:8000/tag \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {
        "media_type": "video",
        "media_path": "/workspaces/ai-toolkit/video.mp4",
        "frame_interval": 0.25,
        "max_frame_count": 15,
        "general_threshold": 0.35,
        "character_threshold": 0.85
      }
    ]
  }'
```

### Tag multiple inputs in one request

```bash
curl -X POST http://localhost:8000/tag \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "/workspaces/ai-toolkit/cat.png",
      {
        "media_path": "/workspaces/ai-toolkit/dog.png",
        "general_threshold": 0.4
      }
    ]
  }'
```

## Tagger Model Management

### Offload tagger model to CPU

```bash
curl -X GET http://localhost:8000/tag/free
```

## Tagging Parameter Reference

### Request Shape

```json
{
  "model_path": "/path/to/wd14-vit-v1",
  "input": [
    "/path/to/image.png",
    {
      "media_url": "https://example.com/sample.png"
    },
    {
      "media_path": "/path/to/video.mp4",
      "media_type": "video",
      "frame_interval": 0.25,
      "max_frame_count": 50,
      "general_threshold": 0.35,
      "character_threshold": 0.85
    }
  ]
}
```

### Input Item Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `media_path` | string | required* | Path to a local image or video file |
| `media_url` | string | optional | URL to an image or video file (use instead of `media_path`) |
| `media_type` | string | inferred | `"image"` or `"video"`; inferred from file extension when omitted |
| `frame_interval` | float | 0.25 | Video frame sampling interval in seconds |
| `max_frame_count` | integer | 50 | Max frames sampled from a video |
| `general_threshold` | float | 0.35 | Minimum confidence for general tags |
| `character_threshold` | float | 0.85 | Minimum confidence for character tags |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | null | Directory containing `model.onnx` and `selected_tags.csv` (or a path to `model.onnx` with `selected_tags.csv` in the same folder) |

### Notes

- `input` must be a list; items can be a string path or a dict with overrides.
- Provide either `media_path` or `media_url`, not both.
- When `model_path` is provided, the server loads from that location instead of downloading.
- Video inference is supported for `.mp4`, `.webm`, `.gifv`, and `.gif` files.
- Responses include `rating` and `tags` only, matching civitai-tagger output.
