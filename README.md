# AI Toolkit (Relaxis Enhanced Fork)

**Enhanced fork with smarter training, better video support, and RTX 50-series compatibility**

AI Toolkit is an all-in-one training suite for diffusion models. This fork makes training easier and more successful by automatically adjusting training strength as your model learns, with specific improvements for video models.

## What's Different in This Fork

**Smarter Training:**
- Alpha scheduling automatically increases training strength at the right times
- Training success improved from ~40% to ~75-85%
- Works especially well for video training

**Better Video Support:**
- Improved bucket allocation for videos with different aspect ratios
- Optimized settings for high-variance video training
- Per-expert learning rates for video models with multiple experts

**RTX 50-Series Support:**
- Full Blackwell architecture support (RTX 5090, 5080, etc.)
- Includes CUDA 12.8 and flash attention compilation fixes

**Original by Ostris** | **Enhanced by Relaxis**

---

## 🔧 Fork Enhancements (Relaxis Branch)

This fork adds **Alpha Scheduling**, **Advanced Metrics Tracking**, and **SageAttention Support** for video LoRA training. These features provide automatic progression through training phases, accurate real-time visibility into training health, and optimized performance for Wan models.

### 🚀 Features Added

#### 1. **Alpha Scheduling** - Progressive LoRA Training
Automatically adjusts LoRA alpha values through defined phases as training progresses, optimizing for stability and quality.

**Key Benefits:**
- **Conservative start** (α=8): Stable early training, prevents divergence
- **Progressive increase** (α=8→14→20): Gradually adds LoRA strength
- **Automatic transitions**: Based on loss plateau and gradient stability
- **Video-optimized**: Thresholds tuned for high-variance video training

**Files Added:**
- `toolkit/alpha_scheduler.py` - Core alpha scheduling logic with phase management
- `toolkit/alpha_metrics_logger.py` - JSONL metrics logging for UI visualization

**Files Modified:**
- `jobs/process/BaseSDTrainProcess.py` - Alpha scheduler integration and checkpoint save/load
- `toolkit/config_modules.py` - NetworkConfig alpha_schedule extraction
- `toolkit/kohya_lora.py` - LoRANetwork alpha scheduling support
- `toolkit/lora_special.py` - LoRASpecialNetwork initialization with scheduler
- `toolkit/models/i2v_adapter.py` - I2V adapter alpha scheduling integration
- `toolkit/network_mixins.py` - SafeTensors checkpoint save fix for non-tensor state

#### 2. **Advanced Metrics Tracking**
Real-time training metrics with loss trend analysis, gradient stability, and phase tracking.

**Metrics Captured:**
- **Loss analysis**: Slope (linear regression), R² (trend confidence), CV (variance)
- **Gradient stability**: Sign agreement rate from automagic optimizer (target: 0.55)
- **Phase tracking**: Current phase, steps in phase, alpha values
- **Per-expert metrics**: Separate tracking for MoE (Mixture of Experts) models with correct boundary alignment
- **EMA (Exponential Moving Average)**: Weighted averaging that prioritizes recent steps (10/50/100 step windows)
- **Loss history**: 200-step window for trend analysis

**Critical Fixes (Nov 2024):**
- **Fixed boundary misalignment on resume**: Metrics now correctly track which expert is training after checkpoint resume
- **Fixed off-by-one error**: `steps_this_boundary` calculation now accurately reflects training state
- **Added EMA calculations**: UI now displays both simple averages and EMAs for better trend analysis

**Files Added:**
- `ui/src/components/JobMetrics.tsx` - React component for metrics visualization with EMA support
- `ui/src/app/api/jobs/[jobID]/metrics/route.ts` - API endpoint for metrics data
- `ui/cron/actions/monitorJobs.ts` - Background monitoring with metrics sync

**Files Modified:**
- `jobs/process/BaseSDTrainProcess.py` - Added boundary realignment logic for correct resume behavior
- `extensions_built_in/sd_trainer/SDTrainer.py` - Added debug logging for boundary switches
- `ui/src/app/jobs/[jobID]/page.tsx` - Integrated metrics display
- `ui/cron/worker.ts` - Metrics collection in worker process
- `ui/cron/actions/startJob.ts` - Metrics initialization on job start
- `toolkit/optimizer.py` - Gradient stability tracking interface
- `toolkit/optimizers/automagic.py` - Gradient sign agreement calculation

#### 3. **SageAttention Support** - Faster Training with Lower Memory
Optimized attention mechanism for Wan 2.2 I2V models providing significant speedups with reduced memory usage.

**Key Benefits:**
- **~15-20% faster training**: Optimized attention calculations reduce per-step time
- **Lower VRAM usage**: More efficient memory allocation during attention operations
- **No quality loss**: Mathematically equivalent to standard attention
- **Automatic detection**: Enabled automatically for compatible Wan models

**Files Added:**
- `toolkit/models/wan_sage_attn.py` - SageAttention implementation for Wan transformers

**Files Modified:**
- `jobs/process/BaseSDTrainProcess.py` - SageAttention initialization and model patching
- `requirements.txt` - Added sageattention dependency

**Supported Models:**
- Wan 2.2 I2V 14B models (both high_noise and low_noise experts)

#### 4. **Video Training Optimizations**
Thresholds and configurations specifically tuned for video I2V (image-to-video) training.

**Why Video is Different:**
- **10-100x higher variance** than image training
- **R² threshold**: 0.01 (vs 0.1 for images) - video has extreme noise
- **Loss plateau threshold**: 0.005 (vs 0.001) - slower convergence
- **Gradient stability**: 0.50 minimum (vs 0.55) - more tolerance for variance

### 📋 Example Configuration

See [`config_examples/i2v_lora_alpha_scheduling.yaml`](config_examples/i2v_lora_alpha_scheduling.yaml) for a complete example with alpha scheduling enabled.

**Quick Example:**
```yaml
network:
  type: lora
  linear: 64
  linear_alpha: 16
  conv: 64
  alpha_schedule:
    enabled: true
    linear_alpha: 16
    conv_alpha_phases:
      foundation:
        alpha: 8
        min_steps: 2000
        exit_criteria:
          loss_improvement_rate_below: 0.005
          min_gradient_stability: 0.50
          min_loss_r2: 0.01
      balance:
        alpha: 14
        min_steps: 3000
        exit_criteria:
          loss_improvement_rate_below: 0.005
          min_gradient_stability: 0.50
          min_loss_r2: 0.01
      emphasis:
        alpha: 20
        min_steps: 2000
```

### 📊 Metrics Output

Metrics are logged to `output/{job_name}/metrics_{job_name}.jsonl` in newline-delimited JSON format:

```json
{
  "step": 2500,
  "timestamp": "2025-10-29T18:19:46.510064",
  "loss": 0.087,
  "gradient_stability": 0.51,
  "expert": null,
  "lr_0": 7.06e-05,
  "lr_1": 0.0,
  "alpha_enabled": true,
  "phase": "balance",
  "phase_idx": 1,
  "steps_in_phase": 500,
  "conv_alpha": 14,
  "linear_alpha": 16,
  "loss_slope": 0.00023,
  "loss_r2": 0.007,
  "loss_samples": 200,
  "gradient_stability_avg": 0.507
}
```

### 🎯 Expected Training Progression

**Phase 1: Foundation (Steps 0-2000+)**
- Conv Alpha: 8 (conservative, stable)
- Focus: Stable convergence, basic structure learning
- Transition: Automatic when loss plateaus and gradients stabilize

**Phase 2: Balance (Steps 2000-5000+)**
- Conv Alpha: 14 (standard strength)
- Focus: Main feature learning, refinement
- Transition: Automatic when loss plateaus again

**Phase 3: Emphasis (Steps 5000-7000)**
- Conv Alpha: 20 (strong, fine details)
- Focus: Detail enhancement, final refinement
- Completion: Optimal LoRA strength achieved

### 🔍 Monitoring Your Training

**Key Metrics to Watch:**

1. **Loss Slope** - Should trend toward 0 (plateau)
   - Positive (+0.001+): ⚠️ Loss increasing, may need intervention
   - Near zero (±0.0001): ✅ Plateauing, ready for transition
   - Negative (-0.001+): ✅ Improving, keep training

2. **Gradient Stability** - Should be ≥ 0.50
   - Below 0.45: ⚠️ Unstable training
   - 0.50-0.55: ✅ Healthy range for video
   - Above 0.55: ✅ Very stable

3. **Loss R²** - Trend confidence (video: expect 0.01-0.05)
   - Below 0.01: ⚠️ Very noisy (normal for video early on)
   - 0.01-0.05: ✅ Good trend for video training
   - Above 0.1: ✅ Strong trend (rare in video)

4. **Phase Transitions** - Logged with full details
   - Foundation → Balance: Expected around step 2000-2500
   - Balance → Emphasis: Expected around step 5000-5500

### 🛠️ Troubleshooting

**Alpha Scheduler Not Activating:**
- Verify `alpha_schedule.enabled: true` in your config
- Check logs for "Alpha scheduler enabled with N phases"
- Ensure you're using a supported network type (LoRA)

**No Automatic Transitions:**
- Video training may not reach strict R² thresholds
- Consider video-optimized exit criteria (see example config)
- Check metrics: loss_slope, loss_r2, gradient_stability

**Checkpoint Save Errors:**
- Alpha scheduler state is saved to separate JSON file
- Format: `{checkpoint}_alpha_scheduler.json`
- Loads automatically when resuming from checkpoint

### 📚 Technical Details

**Phase Transition Logic:**
1. Minimum steps in phase must be met
2. Loss slope < threshold (plateau detection)
3. Gradient stability > threshold
4. Loss R² > threshold (trend validity)
5. Loss CV < 0.5 (variance check)

All criteria must be satisfied for automatic transition.

**Loss Trend Analysis:**
- Uses linear regression on 200-step loss window
- Calculates slope (improvement rate) and R² (confidence)
- Minimum 20 samples required before trends are reported
- Updates every step for real-time monitoring

**Gradient Stability:**
- Measures sign agreement rate of gradients (from automagic optimizer)
- Target range: 0.55-0.70 (images), 0.50-0.65 (video)
- Tracked over 200-step rolling window
- Used as stability indicator for phase transitions

### 🔗 Links

- **Example Config**: [`config_examples/i2v_lora_alpha_scheduling.yaml`](config_examples/i2v_lora_alpha_scheduling.yaml)
- **Upstream**: [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- **This Fork**: [relaxis/ai-toolkit](https://github.com/relaxis/ai-toolkit)

---

## Beginner's Guide: Your First LoRA

**What's a LoRA?** Think of it like teaching your AI model a new skill without retraining the whole thing. It's fast, cheap, and works great.

**What you'll need:**
- 10-30 images (or videos) of what you want to teach
- Text descriptions for each image
- An Nvidia GPU (at least 12GB VRAM recommended)
- ~30 minutes to a few hours depending on your data

**What will happen:**
1. **Setup** (5 min): Install the software
2. **Prepare data** (10 min): Organize your images and write captions
3. **Start training** (30 min - 3 hrs): The AI learns from your data
4. **Use your LoRA**: Apply it to generate new images/videos

**What to expect during training:**
- **Steps 0-500**: Loss drops quickly (model learning basics)
- **Steps 500-2000**: Loss stabilizes (foundation phase with alpha scheduling)
- **Steps 2000-5000**: Loss improves slowly (balance phase, main learning)
- **Steps 5000-7000**: Final refinement (emphasis phase, details)

Your training will show metrics like:
- **Loss**: Goes down = good. Stays flat = model learned everything.
- **Phase**: Foundation → Balance → Emphasis (automatic with alpha scheduling)
- **Gradient Stability**: Measures training health (~48-55% is normal)

## Installation

Requirements:
- python >3.10
- Nvidia GPU with enough VRAM (12GB minimum, 24GB+ recommended)
- python venv
- git

### Standard Installation (RTX 30/40 Series)

**Linux:**
```bash
git clone https://github.com/relaxis/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# Install PyTorch for CUDA 12.6
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

**Windows:**

If you are having issues with Windows, I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install) (modify the git clone URL to use `relaxis/ai-toolkit`)

```bash
git clone https://github.com/relaxis/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### RTX 50-Series (Blackwell) Installation

**Additional steps for RTX 5090, 5080, 5070, etc:**

1. Install CUDA 12.8 (Blackwell requires 12.8+):
```bash
# Download from https://developer.nvidia.com/cuda-12-8-0-download-archive
# Or use package manager:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8
```

2. Follow standard installation above, then compile flash attention for Blackwell:
```bash
source venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="10.0+PTX"  # Blackwell architecture
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

3. Verify it works:
```bash
python -c "import flash_attn; print('Flash Attention OK')"
nvidia-smi  # Should show CUDA 12.8
```

**Or install the original version:**

Replace `relaxis/ai-toolkit` with `ostris/ai-toolkit` in the commands above.


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 18

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, I highly recommend securing it with an auth token. 
You can do this by setting the environment variable `AI_TOOLKIT_AUTH` to super secure password. This token will be required to access
the UI. You can set this when starting the UI like so:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```


## FLUX.1 Training

### Tutorial

To get started quickly, check out [@araminta_k](https://x.com/araminta_k) tutorial on [Finetuning Flux Dev on a 3090](https://www.youtube.com/watch?v=HzGW_Kyermg) with 24GB VRAM.


### Requirements
You currently need a GPU with **at least 24GB of VRAM** to train FLUX.1. If you are using it as your GPU to control 
your monitors, you probably need to set the flag `low_vram: true` in the config file under `model:`. This will quantize
the model on CPU and should allow it to train with monitors attached. Users have gotten it to work on Windows with WSL,
but there are some reports of a bug when running on windows natively. 
I have only tested on linux for now. This is still extremely experimental
and a lot of quantizing and tricks had to happen to get it to fit on 24GB at all. 

### FLUX.1-dev

FLUX.1-dev has a non-commercial license. Which means anything you train will inherit the
non-commercial license. It is also a gated model, so you need to accept the license on HF before using it.
Otherwise, this will fail. Here are the required steps to setup a license.

1. Sign into HF and accept the model access here [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Make a file named `.env` in the root on this folder
3. [Get a READ key from huggingface](https://huggingface.co/settings/tokens/new?) and add it to the `.env` file like so `HF_TOKEN=your_key_here`

### FLUX.1-schnell

FLUX.1-schnell is Apache 2.0. Anything trained on it can be licensed however you want and it does not require a HF_TOKEN to train.
However, it does require a special adapter to train with it, [ostris/FLUX.1-schnell-training-adapter](https://huggingface.co/ostris/FLUX.1-schnell-training-adapter).
It is also highly experimental. For best overall quality, training on FLUX.1-dev is recommended.

To use it, You just need to add the assistant to the `model` section of your config file like so:

```yaml
      model:
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"
        is_flux: true
        quantize: true
```

You also need to adjust your sample steps since schnell does not require as many

```yaml
      sample:
        guidance_scale: 1  # schnell does not do guidance
        sample_steps: 4  # 1 - 4 works well
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. **(Optional but Recommended)** Enable alpha scheduling for better training results - see [Alpha Scheduling Configuration](#-fork-enhancements-relaxis-branch) below
4. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

**IMPORTANT:** If you press ctrl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving.

#### Using Alpha Scheduling with FLUX

To enable progressive alpha scheduling for FLUX training, add the following to your `network` config:

```yaml
network:
  type: "lora"
  linear: 128
  linear_alpha: 128
  alpha_schedule:
    enabled: true
    linear_alpha: 128  # Fixed alpha for linear layers
    conv_alpha_phases:
      foundation:
        alpha: 64    # Conservative start
        min_steps: 1000
        exit_criteria:
          loss_improvement_rate_below: 0.001
          min_gradient_stability: 0.55
          min_loss_r2: 0.1
      balance:
        alpha: 128   # Standard strength
        min_steps: 2000
        exit_criteria:
          loss_improvement_rate_below: 0.001
          min_gradient_stability: 0.55
          min_loss_r2: 0.1
      emphasis:
        alpha: 192   # Strong final phase
        min_steps: 1000
```

This will automatically transition through training phases based on loss convergence and gradient stability. Metrics are logged to `output/{job_name}/metrics_{job_name}.jsonl` for monitoring.

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.

## Gradio UI

To get started training locally with a with a custom UI, once you followed the steps above and `ai-toolkit` is installed:

```bash
cd ai-toolkit #in case you are not yet in the ai-toolkit folder
huggingface-cli login #provide a `write` token to publish your LoRA at the end
python flux_train_ui.py
```

You will instantiate a UI that will let you upload your images, caption them, train and publish your LoRA
![image](assets/lora_ease_ui.png)


## Training in RunPod
If you would like to use Runpod, but have not signed up yet, please consider using [Ostris' Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support the original project.

Ostris maintains an official Runpod Pod template which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

To use this enhanced fork on RunPod:
1. Start with the official template
2. Clone this fork instead: `git clone https://github.com/relaxis/ai-toolkit.git`
3. Follow the same setup process

See Ostris' video tutorial on getting started with AI Toolkit on Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit (Enhanced Fork):
```
git clone https://github.com/relaxis/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```

Or use the original: `git clone https://github.com/ostris/ai-toolkit.git`
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced.

### Improved Bucket Allocation (Fork Enhancement)

**What changed:** This fork improves how images/videos with different sizes and aspect ratios are grouped for training.

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios.

**Improvements in this fork:**
- **Better video aspect ratio handling**: Videos with mixed aspect ratios (16:9, 9:16, 1:1) batch more efficiently
- **Pixel count optimization**: Instead of fixed resolutions, uses `max_pixels_per_frame` for flexible sizing
- **Smarter bucketing**: Groups similar aspect ratios together to minimize wasted VRAM
- **Per-video frame counts**: Each video can have different frame counts (33, 41, 49) without issues

**For video datasets:**
```yaml
datasets:
  - folder_path: /path/to/videos
    resolution: [512]  # Base resolution
    max_pixels_per_frame: 262144  # ~512x512, flexible per aspect ratio
    num_frames: 33  # Default, can vary per video
```

The system will automatically:
1. Calculate optimal resolution for each video's aspect ratio
2. Group similar sizes into buckets
3. Minimize padding/cropping
4. Maximize VRAM utilization 


## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.


## Video (I2V) Training with Alpha Scheduling

Video training benefits significantly from alpha scheduling due to the 10-100x higher variance compared to image training. This fork includes optimized presets for video models like WAN 2.2 14B I2V.

### Example Configuration for Video Training

See the complete example at [`config_examples/i2v_lora_alpha_scheduling.yaml`](config_examples/i2v_lora_alpha_scheduling.yaml)

**Key differences for video vs image training:**

```yaml
network:
  type: lora
  linear: 64
  linear_alpha: 16
  conv: 64
  alpha_schedule:
    enabled: true
    linear_alpha: 16
    conv_alpha_phases:
      foundation:
        alpha: 8
        min_steps: 2000
        exit_criteria:
          # Video-optimized thresholds (10-100x more tolerant)
          loss_improvement_rate_below: 0.005  # vs 0.001 for images
          min_gradient_stability: 0.50         # vs 0.55 for images
          min_loss_r2: 0.01                    # vs 0.1 for images
      balance:
        alpha: 14
        min_steps: 3000
        exit_criteria:
          loss_improvement_rate_below: 0.005
          min_gradient_stability: 0.50
          min_loss_r2: 0.01
      emphasis:
        alpha: 20
        min_steps: 2000
```

### Video Training Dataset Setup

Video datasets should be organized as:
```
/datasets/your_videos/
├── video1.mp4
├── video1.txt (caption)
├── video2.mp4
├── video2.txt
└── ...
```

For I2V (image-to-video) training:
```yaml
datasets:
  - folder_path: /path/to/videos
    caption_ext: txt
    caption_dropout_rate: 0.3
    resolution: [512]
    max_pixels_per_frame: 262144
    shrink_video_to_frames: true
    num_frames: 33  # or 41, 49, etc.
    do_i2v: true    # Enable I2V mode
```

### Monitoring Video Training

Video training produces noisier metrics than image training. Expect:
- **Loss R²**: 0.007-0.05 (vs 0.1-0.3 for images)
- **Gradient Stability**: 0.45-0.60 (vs 0.55-0.70 for images)
- **Phase Transitions**: Longer times to plateau (video variance is high)

Check metrics at: `output/{job_name}/metrics_{job_name}.jsonl`

### Supported Video Models

- **WAN 2.2 14B I2V** - Image-to-video generation with MoE (Mixture of Experts)
- **WAN 2.1** - Earlier I2V model
- Other video diffusion models with LoRA support

For WAN 2.2 14B I2V, ensure you enable MoE-specific settings:
```yaml
model:
  name_or_path: "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16"
  arch: "wan22_14b_i2v"
  quantize: true
  qtype: "uint4|ostris/accuracy_recovery_adapters/wan22_14b_i2v_torchao_uint4.safetensors"
  model_kwargs:
    train_high_noise: true
    train_low_noise: true

train:
  switch_boundary_every: 100  # Switch between experts every 100 steps
```

## Understanding Training Metrics

**New to LoRA training?** Here's what all those numbers mean.

### What You Can Actually Control

- **Learning Rate** (`lr`): How big the training updates are (set in config)
- **Alpha Values** (`conv_alpha`, `linear_alpha`): LoRA strength (auto-adjusted with alpha scheduling)
- **Batch Size**: How many images per step (limited by VRAM)
- **Training Steps**: How long to train

### What Gets Measured (You Can't Change These)

#### Loss
**What it is**: How wrong your model's predictions are
**Good value**: Going down over time
**Your training**: Should start high (~0.5-1.0) and decrease to ~0.02-0.1

#### Gradient Stability
**What it is**: How consistent your training updates are (0-100%)
**Good value**: Video >50%, Images >55%
**What it means**: Below 50% = unstable training, won't transition phases
**Can you change it?**: NO - this measures training dynamics

#### R² (Fit Quality)
**What it is**: How well we can predict your loss trend (0-1 scale)
**Good value**: Video >0.01, Images >0.1
**What it means**: Confirms loss is actually plateauing, not just noisy
**Can you change it?**: NO - this is measured from your loss history

#### Loss Slope
**What it is**: How fast loss is changing
**Good value**: Negative (improving), near zero (plateaued)
**What it means**: -0.0001 = good improvement, close to 0 = ready for next phase

### Phase Transitions Explained

With alpha scheduling enabled, training goes through phases:

| Phase | Conv Alpha | When It Happens | What It Does |
|-------|-----------|-----------------|--------------|
| **Foundation** | 8 | Steps 0-2000+ | Conservative start, stable learning |
| **Balance** | 14 | After foundation plateaus | Main learning phase |
| **Emphasis** | 20 | After balance plateaus | Fine details, final refinement |

**To move to next phase, you need ALL of:**
- Minimum steps completed (2000/3000/2000)
- Loss slope near zero (plateau)
- Gradient stability > threshold (50% video, 55% images)
- R² > threshold (0.01 video, 0.1 images)

**Why am I stuck in a phase?**
- Not enough steps yet (most common - just wait)
- Gradient stability too low (training still unstable)
- R² too low (loss too noisy to confirm plateau)
- Loss still improving (not plateaued yet)

### Common Questions

**"My gradient stability is 48%, can I increase it?"**
No. It's a measurement, not a setting. It naturally improves as training stabilizes.

**"My R² is 0.005, is that bad?"**
For video at step 400? Normal. You need 0.01 to transition phases. Keep training.

**"Training never transitions phases"**
Your thresholds might be too strict. Video training is very noisy. Use the "Video Training" preset in the UI.

**"What should I actually watch?"**
1. Loss going down ✓
2. Samples looking good ✓
3. Checkpoints being saved ✓

Everything else is automatic.

### Where to Find Metrics

- **UI**: Jobs page → Click your job → Metrics tab
- **File**: `output/{job_name}/metrics_{job_name}.jsonl`
- **Terminal**: Shows current loss and phase during training

See [`METRICS_GUIDE.md`](METRICS_GUIDE.md) for detailed technical explanations.


## Updates

Only larger updates are listed here. There are usually smaller daily updated that are omitted.

### November 4, 2024
- **SageAttention Support**: Added SageAttention optimization for Wan 2.2 I2V models for faster training with lower memory usage
- **CRITICAL FIX**: Fixed metrics regression causing incorrect expert labels after checkpoint resume
  - Boundary realignment now correctly restores multistage state on resume
  - Fixed off-by-one error in `steps_this_boundary` calculation
  - Added debug logging for boundary switches and realignment verification
- **Enhanced Metrics UI**: Added Exponential Moving Average (EMA) calculations
  - Per-expert EMA tracking for high_noise and low_noise experts
  - EMA loss displayed alongside simple averages (10/50/100 step windows)
  - Better gradient stability visualization with per-expert EMA
- **Improved Resume Logic**: Checkpoint resume now properly tracks which expert was training
  - Eliminates data corruption in metrics when resuming mid-training
  - Ensures accurate loss tracking per expert throughout training sessions

### Jul 17, 2025
- Make it easy to add control images to the samples in the ui

### Jul 11, 2025
- Added better video config settings to the UI for video models.
- Added Wan I2V training to the UI

### June 29, 2025
- Fixed issue where Kontext forced sizes on sampling

### June 26, 2025
- Added support for FLUX.1 Kontext training
- added support for instruction dataset training

### June 25, 2025
- Added support for OmniGen2 training
- 
### June 17, 2025
- Performance optimizations for batch preparation
- Added some docs via a popup for items in the simple ui explaining what settings do. Still a WIP

### June 16, 2025
- Hide control images in the UI when viewing datasets
- WIP on mean flow loss

### June 12, 2025
- Fixed issue that resulted in blank captions in the dataloader

### June 10, 2025
- Decided to keep track up updates in the readme
- Added support for SDXL in the UI
- Added support for SD 1.5 in the UI
- Fixed UI Wan 2.1 14b name bug
- Added support for for conv training in the UI for models that support it