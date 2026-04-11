# Ostris AI Toolkit

AI Toolkit is an easy to use all in one training suite for diffusion models. I try to support all the latest models on consumer grade hardware. Image and video models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable. Free and open source.



## Supported Models

### Image
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)

### Instruction / Edit
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)

### Video
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)

### Audio
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)

### Experimental
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)

## Installation

Requirements:
- python >=3.10 (3.12 recommended)
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt
```

For devices running **DGX OS** (including DGX Spark), follow [these](dgx_instructions.md) instructions.


Windows:

If you are having issues with Windows. I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install)

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

MacOS:

Experimental support for Silicon Macs is available. I do not have a Mac with enough RAM to fully test this
so please let me know if there are issues. There is a convience script to install and run on MacOS 
locates at `./run_mac.zsh` that will install the dependencies locally and run the UI. To run this, 
do the following:

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
chmod +x run_mac.zsh
./run_mac.zsh
```


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 20

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

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

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
If you would like to use Runpod, but have not signed up yet, please consider using [my Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support this project.


I maintain an official Runpod Pod template here which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

I have also created a short video showing how to get started using AI Toolkit with Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
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

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 


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


## Support My Work

If you enjoy my projects or use them commercially, please consider sponsoring me. Every bit helps! 💖

<a href="https://ostris.com/sponsors" target="_blank"><img src="https://ostris.com/wp-content/uploads/2025/05/support-banner2.png" alt="Support my work" style="max-width:100%;height:auto;"></a>

### Current Sponsors

All of these people / organizations are the ones who selflessly make this project possible. Thank you!!

_Last updated: 2026-03-31 18:10 UTC_

<p align="center">
<a href="https://x.com/NuxZoe" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1919488160125616128/QAZXTMEj_400x400.png" alt="a16z" width="275" height="275" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/replicate" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/60410876?v=4" alt="Replicate" width="275" height="275" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/huggingface" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/25720743?v=4" alt="Hugging Face" width="275" height="275" style="border-radius:8px;margin:5px;display: inline-block;"></a>
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<a href="https://www.pixelcut.ai/" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1496882159658885133/11asz2Sc_400x400.jpg" alt="Pixelcut" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/weights-ai" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/185568492?v=4" alt="Weights" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/josephrocca" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/1167575?u=92d92921b4cb5c8c7e225663fed53c4b41897736&v=4" alt="josephrocca" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/93304/J" alt="Joseph Rocca" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/161471720/dd330b4036d44a5985ed5985c12a5def/eyJ3IjoyMDB9/1.jpeg?token-hash=k1f4Vv7TevzYa9tqlzAjsogYmkZs8nrXQohPCDGJGkc%3D" alt="Vladimir Sotnikov" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/33158543/C" alt="clement Delangue" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/8654302/b0f5ebedc62a47c4b56222693e1254e9/eyJ3IjoyMDB9/2.jpeg?token-hash=suI7_QjKUgWpdPuJPaIkElkTrXfItHlL8ZHLPT-w_d4%3D" alt="Misch Strotz" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://www.runcomfy.com/trainer/ai-toolkit/app" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1747828425736273922/nlPQTDYO_400x400.jpg" alt="RunComfy" width="200" height="200" style="border-radius:8px;margin:5px;display: inline-block;"></a>
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/120239481/49b1ce70d3d24704b8ec34de24ec8f55/eyJ3IjoyMDB9/1.jpeg?token-hash=o0y1JqSXqtGvVXnxb06HMXjQXs6OII9yMMx5WyyUqT4%3D" alt="nitish PNR" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/2298192/1228b69bd7d7481baf3103315183250d/eyJ3IjoyMDB9/1.jpg?token-hash=opN1e4r4Nnvqbtr8R9HI8eyf9m5F50CiHDOdHzb4UcA%3D" alt="Mohamed Oumoumad" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/548524/S" alt="Steve Hanff" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/169502989/220069e79ce745b29237e94c22a729df/eyJ3IjoyMDB9/1.png?token-hash=E8E2JOqx66k2zMtYUw8Gy57dw-gVqA6OPpdCmWFFSFw%3D" alt="Timothy Bielec" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/9547341/bb35d9a222fd460e862e960ba3eacbaf/eyJ3IjoyMDB9/1.jpeg?token-hash=Q2XGDvkCbiONeWNxBCTeTMOcuwTjOaJ8Z-CAf5xq3Hs%3D" alt="Travis Harrington" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/5021048/c6beacab0fdb4568bf9f0d549aa4bc44/eyJ3IjoyMDB9/1.jpeg?token-hash=JTEtFVzUeU7pQw4R3eSn6rGgqgi44uc2rDBAv6F6A4o%3D" alt="Infinite " width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/33228112/J" alt="Jimmy Simmons" width="150" height="150" style="border-radius:8px;margin:5px;display: inline-block;">
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<img src="https://c8.patreon.com/4/200/55206617/X" alt="xv" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/80767260/1fa7b3119f9f4f40a68452e57de59bfe/eyJ3IjoyMDB9/1.jpeg?token-hash=H34Vxnd58NtbuJU1XFYPkQnraVXSynZHSL3SMMcdKbI%3D" alt="nuliajuk" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/40761075/R" alt="Randy McEntee" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/27288932/6c35d2d961ee4e14a7a368c990791315/eyJ3IjoyMDB9/1.jpeg?token-hash=TGIto_PGEG2NEKNyqwzEnRStOkhrjb3QlMhHA3raKJY%3D" alt="David Garrido" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/E2GO" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/1776669?u=bf52b2691fa7d1e421d6167b804a2c1cf3b229e7&v=4" alt="E2GO" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/128354277/52c073d323924b02ada90c9eacc6b0a0/eyJ3IjoyMDB9/1.png?token-hash=Oc0mVzELN1s1r0lLQTEO_sfJ2lEMC3X-By2O2bG6h_Q%3D" alt="Alastair Green" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/7208949/D" alt="D G" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/358350/L" alt="L D" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/179944/P" alt="Paul Kroll" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://x.com/NuxZoe" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1916482710069014528/RDLnPRSg_400x400.jpg" alt="tungsten" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/3712451/432e22a355494ec0a1ea1927ff8d452e/eyJ3IjoyMDB9/7.jpeg?token-hash=OpQ9SAfVQ4Un9dSYlGTHuApZo5GlJ797Mo0DtVtMOSc%3D" alt="David Shorey" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/squewel" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/97603184?v=4" alt="squewel" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://clwill.com/" target="_blank" rel="noopener noreferrer"><img src="https://images.squarespace-cdn.com/content/v1/63d444727a5d5f304f89eebe/c9def9ce-3824-404d-a8bb-96b6236338ca/favicon.ico?format=100w" alt="Christopher Williams" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="http://www.ir-ltd.net" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/1602579392198283264/6Tm2GYus_400x400.jpg" alt="IR-Entertainment Ltd" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Alexander Korchemniy" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/50279373/326f5dc32cc749d7afb8df64f202ad00/eyJ3IjoyMDB9/1.jpeg?token-hash=PUJrhne0p1Z-DIKb6_NV7ZI7su5EknTeejjBCffg0IQ%3D" alt="Jürgen Stein" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/49347178/8cd9db18638c4b9d8ec90ccf729d6704/eyJ3IjoyMDB9/1.jpeg?token-hash=zw9cDUwUupmEAMLeQ8AScBOt8p2mkdbQGXU6PS4j4zk%3D" alt="Khoi Nguyen" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/98811435/3a3632d1795b4c2b9f8f0270f2f6a650/eyJ3IjoyMDB9/1.jpeg?token-hash=657rzuJ0bZavMRZW3XZ-xQGqm3Vk6FkMZgFJVMCOPdk%3D" alt="EmmanuelMr18" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/27791680/J" alt="Jean-Tristan Marin" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/93348210/5c650f32a0bc481d80900d2674528777/eyJ3IjoyMDB9/1.jpeg?token-hash=0jiknRw3jXqYWW6En8bNfuHgVDj4LI_rL7lSS4-_xlo%3D" alt="Armin Behjati" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/155963250/D" alt="Drama Labs GmbH" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/Slartibart23" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/133593860?u=31217adb2522fb295805824ffa7e14e8f0fca6fa&v=4" alt="Slarti" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/570742/4ceb33453a5a4745b430a216aba9280f/eyJ3IjoyMDB9/1.jpg?token-hash=nPcJ2zj3sloND9jvbnbYnob2vMXRnXdRuujthqDLWlU%3D" alt="Al H" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/82763/f99cc484361d4b9d94fe4f0814ada303/eyJ3IjoyMDB9/1.jpeg?token-hash=A3JWlBNL0b24FFWb-FCRDAyhs-OAxg-zrhfBXP_axuU%3D" alt="Doron Adler" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/99036356/7ae9c4d80e604e739b68cca12ee2ed01/eyJ3IjoyMDB9/3.png?token-hash=ZhsBMoTOZjJ-Y6h5NOmU5MT-vDb2fjK46JDlpEehkVQ%3D" alt="njgnfhahfnhnwir" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/141098579/1a9f0a1249d447a7a0df718a57343912/eyJ3IjoyMDB9/2.png?token-hash=_n-AQmPgY0FP9zCGTIEsr5ka4Y7YuaMkt3qL26ZqGg8%3D" alt="The Local Lab" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/53077895/M" alt="Marc" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/30931983/54ab4e4ceab946e79a6418d205f9ed51/eyJ3IjoyMDB9/1.png?token-hash=j2phDrgd6IWuqKqNIDbq9fR2B3fMF-GUCQSdETS1w5Y%3D" alt="HestoySeghuro ." width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/4105384/J" alt="Jack Blakely" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/103077711/bb215761cc004e80bd9cec7d4bcd636d/eyJ3IjoyMDB9/2.jpeg?token-hash=3U8kdZSUpnmeYIDVK4zK9TTXFpnAud_zOwBRXx18018%3D" alt="John Dopamine" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/46680573/ee3d99c04a674dd5a8e1ecfb926db6a2/eyJ3IjoyMDB9/1.jpeg?token-hash=cgD4EXyfZMPnXIrcqWQ5jGqzRUfqjPafb9yWfZUPB4Q%3D" alt="Neil Murray" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/44568304/a9d83a0e786b41b4bdada150f7c9271c/eyJ3IjoyMDB9/1.jpeg?token-hash=FtxnwrSrknQUQKvDRv2rqPceX2EF23eLq4pNQYM_fmw%3D" alt="Albert Bukoski" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/5048649/B" alt="Ben Ward" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/134129880/680c7e14cd1a4d1a9face921fb010f88/eyJ3IjoyMDB9/1.png?token-hash=5fqqHE6DCTbt7gDQL7VRcWkV71jF7FvWcLhpYl5aMXA%3D" alt="Bharat Prabhakar" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/494309/J" alt="Julian Tsependa" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/111904990/08b1cf65be6a4de091c9b73b693b3468/eyJ3IjoyMDB9/1.png?token-hash=_Odz6RD3CxtubEHbUxYujcjw6zAajbo3w8TRz249VBA%3D" alt="Brian Smith" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/5602036/c7b6e02bab1241fc83ff5a0cedf19b43/eyJ3IjoyMDB9/1.jpeg?token-hash=nnd10QRNxqaHmhwr-zQh4EIlBDIFJEvt65YB3ebjhNw%3D" alt="Kelevra Quackenstien" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/159203973/36c817f941ac4fa18103a4b8c0cb9cae/eyJ3IjoyMDB9/1.png?token-hash=zkt72HW3EoiIEAn3LSk9gJPBsXfuTVcc4rRBS3CeR8w%3D" alt="Marko jak" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/11198131/e696d9647feb4318bcf16243c2425805/eyJ3IjoyMDB9/1.jpeg?token-hash=c2c2p1SaiX86iXAigvGRvzm4jDHvIFCg298A49nIfUM%3D" alt="Nicholas Agranoff" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/785333/bdb9ede5765d42e5a2021a86eebf0d8f/eyJ3IjoyMDB9/2.jpg?token-hash=l_rajMhxTm6wFFPn7YdoKBxeUqhdRXKdy6_8SGCuNsE%3D" alt="Sapjes " width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/76566911/6485eaf5ec6249a7b524ee0b979372f0/eyJ3IjoyMDB9/1.jpeg?token-hash=mwCSkTelDBaengG32NkN0lVl5mRjB-cwo6-a47wnOsU%3D" alt="the biitz" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/83034/W" alt="william tatum" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/julien-blanchon" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/11278197?v=4" alt="Blanchon" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/88567307/E" alt="el Chavo" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/117569999/55f75c57f95343e58402529cec852b26/eyJ3IjoyMDB9/1.jpeg?token-hash=squblHZH4-eMs3gI46Uqu1oTOK9sQ-0gcsFdZcB9xQg%3D" alt="James Thompson" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/84873332/H" alt="Htango2" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Frank Vance" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://x.com/RalFingerLP" target="_blank" rel="noopener noreferrer"><img src="https://pbs.twimg.com/profile_images/919595465041162241/ZU7X3T5k_400x400.jpg" alt="RalFinger" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/63510241/A" alt="Andrew Park" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/Spikhalskiy" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/532108?u=2464983638afea8caf4cd9f0e4a7bc3e6a63bb0a&v=4" alt="Dmitry Spikhalsky" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/dylanzonix" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/167351340?v=4" alt="Dylan" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Gary Joseph" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/jakeblakeley" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/2407659?u=be0bc786663527f2346b2e99ff608796bce19b26&v=4" alt="Jake Blakeley" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://pbs.twimg.com/profile_images/445246812723503104/mX9BVPMv_400x400.png" alt="q5sys" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Sylvain Fayette" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/604598/5b0c3030a62d4606848f9ebc1f4318f2/eyJ3IjoyMDB9/1.jpeg?token-hash=EnSp4F3aafnQ9SONb1YrSIQRlQPk29h4TWcRzPUv6-c%3D" alt="Tri3Ax " width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/28533016/e8f6044ccfa7483f87eeaa01c894a773/eyJ3IjoyMDB9/2.png?token-hash=ak-h3JWB50hyenCavcs32AAPw6nNhmH2nBFKpdk5hvM%3D" alt="William Tatum" width="100" height="100" style="border-radius:8px;margin:5px;display: inline-block;">
</p>
<hr style="width:100%;border:none;height:2px;background:#ddd;margin:30px 0;">
<p align="center">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/91298241/1b1e6d698cde4faaaae6fc4c2d95d257/eyJ3IjoyMDB9/1.jpeg?token-hash=GCo7gAF_UUdJqz3FsCq8p1pq3AEoRAoC6YIvy5xEeZk%3D" alt="Daniel Partzsch" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/59408413/a0530a7770b6444bafdf0bc9f589eff0/eyJ3IjoyMDB9/1.jpg?token-hash=BlbxZsQpgchtqjByDuW9T8NoFWmCor5sWI0umhUKNlA%3D" alt="ByteC" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/11180426/J" alt="jarrett towe" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/63232055/2300b4ab370341b5b476902c9b8218ee/eyJ3IjoyMDB9/1.png?token-hash=R9Nb4O0aLBRwxT1cGHUMThlvf6A2MD5SO88lpZBdH7M%3D" alt="Marek P" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/55160464/42d4719ba0834e5d83aa989c04e762da/eyJ3IjoyMDB9/1.jpeg?token-hash=_twZUkW3NREIxGUOWskUdvuZQGEcRv9XMfu5NrnCe5M%3D" alt="Chris Canterbury" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/63920575/D" alt="Dutchman5oh" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/27580949/97c7dd2456a34c71b6429612a9e20462/eyJ3IjoyMDB9/1.jpeg?token-hash=cASxwWk8joAXx4tUAHch5CvTiYBR2UOHMeJK6se5fl0%3D" alt="Gergely Madácsi" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/Wallawalla47" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/46779408?v=4" alt="Ian R" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/33866796/7fd2a214fd5c4062b0dd63a29f8de5bd/eyJ3IjoyMDB9/1.png?token-hash=8s-7yi8GawIlqr0FCTk5JWKy26acMiYlOD8LAk2HqqU%3D" alt="James" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/84891403/83682a2a2d3b49ba9d28e7221edd5752/eyJ3IjoyMDB9/1.jpeg?token-hash=LVB6lta4BonhfPwSUnZIDmSW3IU-eEO4sXD7NSK367g%3D" alt="Koray Birand" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/27667925/6dac043a087e4c498e842dfad193baae/eyJ3IjoyMDB9/1.jpeg?token-hash=0bSVQo7QMMdGxFazeM099gsR0wtf28_ZTXeLIHEbIVk%3D" alt="S.Hasan Rizvi" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/31613309/434500d03f714dc18049306ed3f0165c/eyJ3IjoyMDB9/1.jpg?token-hash=acILbq09wxUfJe-G2nMYUYkvHJ88ZxkzU4JebRPw2P0%3D" alt="Theta Graphics" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/10876902/T" alt="Tyssel" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/5155933/C" alt="Chris Dermody" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/44200812/f84fd628abb243bbaded4203761aca29/eyJ3IjoyMDB9/1.png?token-hash=ArthznCCT4BqOSMj_9oP4ECWWHnrb8nYPUDZ6DqSvMU%3D" alt="kingroka" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/mertguvencli" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/29762151?u=bffbb3564ff18f22d8876c3109bb9f96e6d9d9a8&v=4" alt="Mert Guvencli" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/5233761/N" alt="Newtown " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/82707622/3f0de2ffd6eb4074ba91e81381146e1c/eyJ3IjoyMDB9/1.jpeg?token-hash=wk6wjILO2dDHJla7gn3MH9mEKl08e7PuBDwZRUtEQAw%3D" alt="Russell Norris" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/2986571/S" alt="stev " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Gage Siuniak" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/2888571/65c717bd8a564e469c25aa5858f9821b/eyJ3IjoyMDB9/1.png?token-hash=zwMOgNEoC9hlr2KamiB7TG004gCfJ2exSRDO4dhxo5Q%3D" alt="Derrick Schultz" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/16145383/eaf99f01440d4d1a831584f2d3ab1a2c/eyJ3IjoyMDB9/2.jpg?token-hash=BhictNJpGdyywzEepZrGlEY2anNZZjLDQoo2drXM13o%3D" alt="Gribbly" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/14767188/1f22bccbf86b45a2b32642c3f5a493b3/eyJ3IjoyMDB9/1.png?token-hash=cJhOEsMXSv_d5fcqCu8Q_idyYtqc4UocsOaTflsSmT8%3D" alt="Kukee" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/138787313/c809120005024afa959231fe8b253fd9/eyJ3IjoyMDB9/1.png?token-hash=O6x0kkR4uKBsg_OODFHjZqwAupVztiZEOiXYF_7yKxM%3D" alt="Metryman55" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/zappazack" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/74406132?u=356e66c964f9ca4859b274ff6788aebd16e218d4&v=4" alt="zappazack" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/5752417/G" alt="Guillaume Roy" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/154134231/5d307160968b4c29922e2729bb555c99/eyJ3IjoyMDB9/1.jpeg?token-hash=dNP94e42G_A9CHO5zYfUunS2K80y3BPDHQ3NdzphNRY%3D" alt="Colin Boyd" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/122373805/d0d995f2a7d6483cbbe0e9b14391d1ed/eyJ3IjoyMDB9/1.png?token-hash=oQCZooskREZOB36TW0KNZASDeLc88yswNzF-PqcVQyw%3D" alt="DavidO" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/45804549/8117b86a8c4145348ed392d3ea8c9dde/eyJ3IjoyMDB9/2.png?token-hash=ej_ln6ecs0-Cija3vrXaWYFFyWEK2TWmItJE5ALWP4s%3D" alt="Jadev1311" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/194433979/a18cf671feef435c9a93080f11cc8cf3/eyJ3IjoyMDB9/1.png?token-hash=TN6zMy2-V1Wg5uSpZHstYAZAdb_DYk9Erk3XDjE8--M%3D" alt="Cyril Diagne" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/94453070/S" alt="Speedy2023" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Karl Brewer" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://www.youtube.com/@happyme7055" target="_blank" rel="noopener noreferrer"><img src="https://yt3.googleusercontent.com/ytc/AIdro_mFqhIRk99SoEWY2gvSvVp6u1SkCGMkRqYQ1OlBBeoOVp8=s160-c-k-c0x00ffffff-no-rj" alt="Marcus Rass" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Rainer Kulow" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Stavros Glezakos" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Stavros Glezakos" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/14930909/G" alt="Geno Machino" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/58082790/5f425b9f949047f78d9ae98e86faad35/eyJ3IjoyMDB9/1.png?token-hash=WYfg_M7cLsY-crrv71jcy6LLV77bB0_uD2_aw2f9nJ0%3D" alt="Greg Lemons" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/7436837/K" alt="Ken Finlayson" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/75353/cff7a01bb97a45bba9023f1ff4a5f07a/eyJ3IjoyMDB9/1.jpeg?token-hash=3TxvQTWQSYWeqK4Elb6lX9y5ts21jh5jsWa1cXykcG8%3D" alt="Kenneth Loebenberg" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/31096978/f36222d290d2438cba8cfa3de63453c9/eyJ3IjoyMDB9/1.JPG?token-hash=0gwLI-GVquqxBj3FRR4XqJuRonvT5FsN5rdND2jApL0%3D" alt="Le_Fourbe " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/93681621/d638ff4a9e0a40a7bc2c24bae4d6f353/eyJ3IjoyMDB9/1.png?token-hash=AxFFly1YYJskPzdkaU_M5jgyb0kZijSxB1Yb2AbE9h0%3D" alt="Manuel2Santos" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/4544036/O" alt="Osman Bayazit" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/207918979/3bad9c99fdaa43e89631613e71df21a5/eyJ3IjoyMDB9/1.png?token-hash=SjMA1T2FnOrTymN6MYsO8u4ySPV2qXHCW-bQfX_t_X8%3D" alt="Patrick Gallagher" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/188726649/6db3706d63f14468a58535ae5fd1344c/eyJ3IjoyMDB9/1.png?token-hash=QzCqu543VaxIuxyXo_1qrYqBQAyOhprcfNfNSIN3TYk%3D" alt="Phil Ring" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/25199293/e967e5c4ed884f07b705271e253fd584/eyJ3IjoyMDB9/2.png?token-hash=HXM0U96bf454jUiA6xkGU1tWDOholWDApdSbSaz599U%3D" alt="Rob113" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/2622685/bddc4b42c82c47d8b30b05c000b8127b/eyJ3IjoyMDB9/1.jpg?token-hash=4tEFL9DP2L5dpg7rxUcFBlw27qnHO2ceyG38RtI9_Hg%3D" alt="Saftle " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/7408850/e90af02547724fc59ca1f21565df93b1/eyJ3IjoyMDB9/2.jpg?token-hash=-3gTcxS601y5DbEgVUl1qJh_Tqalv8YfJBAy7Qu68F0%3D" alt="Virtamouse" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/107652364/5cae258ff5cd4c9a8e104861e63d5180/eyJ3IjoyMDB9/1.png?token-hash=qkRK53prBXDFG4b_Opnb80wcvWj6q0FjgNqPoSz24yU%3D" alt="Yi Chen" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/2697420/C" alt="Craig Penn" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Christopher Frey" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="keonmin lee" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="yvggeniy romanskiy" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/76956764/68082831372d4b58b21c87c2d6f81e93/eyJ3IjoyMDB9/1.jpeg?token-hash=_jMdHYevH1sM7a0hPsqpkkupuIGaDvAmkr8stWmpsUw%3D" alt="Andrey Sorokin" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/62933094/d69149b4cb9043e99614d2151c4d1288/eyJ3IjoyMDB9/1.jpg?token-hash=oJSs1KuWe9zorODOtGKn6ceSDjsmOZ4hrohVQ2Y45nQ%3D" alt="Blane" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/15407925/B" alt="Brian M" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/73708729/52866102958248c19e646b6b62c7c51a/eyJ3IjoyMDB9/1.png?token-hash=S_haqcc-5zBK1tefXbphLzvA-MGtmstPNlaHch3k4zo%3D" alt="Cora Nox" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/12844508/fd08528fbed74a359acb1f8d06181c0c/eyJ3IjoyMDB9/1.jpeg?token-hash=TNDGh5TSWmlteKxsvB6FLE9wwawPMyvNBaim2U2KRC4%3D" alt="Dave Talbott" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/195837329/a136ba74b4d94df3a2b37e944beb6b9d/eyJ3IjoyMDB9/1.png?token-hash=oAIpcAmkts3GjjTjJVg2QrYs4UdcXgbW8q11p4kjVqQ%3D" alt="Greg Richards" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/12128150/J" alt="Joshua Genke" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/97609519/M" alt="Mollie" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/106121692/060eb9f09ecc4dceb7fa0a6d3c330b85/eyJ3IjoyMDB9/1.jpeg?token-hash=K6vA5Foyh9tAy3yzCtuYKDRF9McrCbQaEUC61x2x1Ic%3D" alt="Pablo Fonseca" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/rickrender" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/121735855?u=a8187fe40cec7f3afdd7c4bb128e0cca500fc220&v=4" alt="renderartist" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/45125613/8a45d1081bfc43b0bf4cb523558cab65/eyJ3IjoyMDB9/3.jpeg?token-hash=iUZhvndnfAiT97FacklmB4XvnMxj0pvepaHsU7JBxLg%3D" alt="Tiny Tsuruta" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/134029856/d1c895bf165149f69ad81ac426e617e9/eyJ3IjoyMDB9/1.jpeg?token-hash=FPzyMI3pAjnZmRlH_nmy2baIRcGKtQrDnN6aMCOHVwo%3D" alt="v33ts" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Joakim Sällström" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/15703526/L" alt="Leo " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/128259665/7e7627e6442141cdbb8b3a32e590fe5d/eyJ3IjoyMDB9/1.jpeg?token-hash=cnTHMo5sfgLnxVek5QvWEyLBUTmEdLaKcs_8AJbVfbc%3D" alt="Bennett Waisbren" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/24216005/ac538de1daa04619810352e62cd962ec/eyJ3IjoyMDB9/2.jpeg?token-hash=VqZ4vz2lfvrB85QNUng-OB7HLmGZ8Yp85Ay7xCb7xsM%3D" alt="Brian Buie" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/caleboleary" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/12816579?u=d7f6ec4b7caf3c4535385a5fa3d7c155057ef664&v=4" alt="Caleb O'Leary" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/129159473/d6547bf609f24fc486b8a72de925acad/eyJ3IjoyMDB9/1.png?token-hash=SajmmmA4r5PcVkkocZb78TA1MD0HzwHApTy4CJmwOCc%3D" alt="Dustin Lausch" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/91434404/G" alt="GameChanger" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/24294031/16731f3bdacb4a1cb987ec7636e08213/eyJ3IjoyMDB9/1.jpg?token-hash=Df1rhYbhEwtrff3hKbn-lflr1ZDp_KtvDzW4GrBisw8%3D" alt="H.W.Prinz" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/77680244/5e14634dac41465780c28f53b1d6b9d6/eyJ3IjoyMDB9/1.png?token-hash=5TMuHTgcLFmFlJK-TNUEIywxwwYXv2y1kZNDCibgePU%3D" alt="james salanitri" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/2361841/J" alt="Jason Briney" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/774502/902857342f834c08a68a7b13b554e078/eyJ3IjoyMDB9/2.jpeg?token-hash=usMsTs8b58b1mJR9PQhM9KsuU1eewl6B90oWRuyaWDI%3D" alt="Wolkenfels " width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/181113408/59fb8db40ca944c2897a295dcfed7340/eyJ3IjoyMDB9/1.png?token-hash=E9iFUUk_Q0cV0gkbiLhLkKwvgPhHTdvalcQsE9hLfd4%3D" alt="John D" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/lirexxx" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/94787562?u=ed7e681cbc200269a081c4151d6adfa6ef728f85&v=4" alt="Dimitar A." width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/157177642/cac4925553f74deb9f9285781839fba8/eyJ3IjoyMDB9/1.png?token-hash=osNeLRXRgvuWKAviRBcPjHzWJFh61MdRtjVgivdeZl0%3D" alt="R132" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Heikki Rinkinen" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Josh Lindo" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Michael Styne" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Michael Styne" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Phan Dao" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="StrictLine e.U." width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="The Rope Dude" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Till Meyer" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Valarm, LLC" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Valarm, LLC" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/204926456/739abd8abc2f4deb965fdacfb5bd7edf/eyJ3IjoyMDB9/1.jpeg?token-hash=ALGKzAFFxxFmwKGb44pmH8A-9sjUPJQEIXXjmWdWIw0%3D" alt="Mal Mallabar" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Xavier Climent" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/76554725/M" alt="Moritz Hutten" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/137558630/04e82f23d76b4e049102529b2ae4693f/eyJ3IjoyMDB9/1.jpeg?token-hash=aA2XcIi-yQske0sUj-L_X4ASuCLRWCBFaAmvUKqaMY8%3D" alt="AAYUSH BHADANI" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/29010107/37b05d32281f460baa28b4a2d5f8dd52/eyJ3IjoyMDB9/3.jpg?token-hash=5FngEN5rK-hCAgHUM0EybhMTuHwRZI1gbbZyntuuH6g%3D" alt="Adel Gamal" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/42375181/a8d4b6dd849c47d596ba1d49e165b658/eyJ3IjoyMDB9/1.jpeg?token-hash=vmkkWAHO-Vv-drVE3JpiLd9MquixdYnV0pxhKmay0AU%3D" alt="Charles Blakemore" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/148105261/20988aa43bec4c38ad1293cfd7c8677f/eyJ3IjoyMDB9/1.png?token-hash=YZp1Sdn13WFKXLlJMtSxdjrJ7aHmo15-PbKD7DcBzmU%3D" alt="Chris Williams" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/claygraffix" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/1283083?v=4" alt="claygraffix" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="David Hooper" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/15533741/D" alt="Dfence" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/201405121/7fc9ef651c6b4a8faea2acd2d1a82cae/eyJ3IjoyMDB9/1.jpeg?token-hash=Q-ACM_hIPVWRfd5CKGl2qrzoHb5Mh5PARNAyKjtZcV0%3D" alt="Evan Forster" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/36227336/2d5602535ca64301a5555c7c027042c6/eyJ3IjoyMDB9/1.jpeg?token-hash=EglM8DWBx6fMiL_9oOJddZCTYYlpv07jL0OVhxsI7Rk%3D" alt="Greg Abousleman" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Jean-Paul Lerault" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Rudolf Goertz" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/ShinChven" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/3351486?u=a70586ea24bb3acadab3019083e78500ddeab641&v=4" alt="ShinChven ✨" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Tommy Falkowski" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Victor-Ray Valdez" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/Jefferderp" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/13530594?v=4" alt="Jefferderp" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/ekgreen7" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/65423214?v=4" alt="ekgreen7" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<a href="https://github.com/marksverdhei" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/46672778?u=d1ba8b17516e6ecf1cd55ca4db2b770f82285aad&v=4" alt="Markus / Mark" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Alex Kovalchuk" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Florian Fiegl" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Kai Buddensiek" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Karol Stępień" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="manuel landron" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Paul Vu Nguyen" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/172131105/4282ce3e9e76458ba76e17bc360411bc/eyJ3IjoyMDB9/1.png?token-hash=8jvSz43m5_KKLX3EqSzt_r5IUTaHokAQ5Uey8-MPDuQ%3D" alt="Jamie Colpitts" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/4420647/A" alt="Alchemist" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/206115527/fef4d02c0fc040059cecdca83ce1008c/eyJ3IjoyMDB9/1.jpeg?token-hash=tCTBHLLM98e6CfqtcsM5BPyqOAW6s8ruhZc7nH3nYRg%3D" alt="Andrew Gould" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c8.patreon.com/4/200/936957/J" alt="Jeroen Van Harten" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/201176148/c9d8944398214a8fa4fefc8fea1e539a/eyJ3IjoyMDB9/1.jpeg?token-hash=a7K3OIIMtyAVp0J76n0Mi-Gcfr_SGMARRdQpcvjL7UY%3D" alt="Kevin Metz" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/114529961/3fe7f48a6dfa4299a7f3184274d9ae2a/eyJ3IjoyMDB9/1.png?token-hash=ye53KqiA6UZO_X8UYF1MoR7VfNV85CuxEP53a3fMF80%3D" alt="Patreon2000" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://c10.patreonusercontent.com/4/patreon-media/p/user/112552091/0771db412e9048ff8856b6a0b29f9ddd/eyJ3IjoyMDB9/1.jpeg?token-hash=dM-UpUK38SHahEPwpRmqTRKlZb55J6XTYRQDm3HnOG0%3D" alt="Paul Bergen" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<a href="https://github.com/ProPatte" target="_blank" rel="noopener noreferrer"><img src="https://avatars.githubusercontent.com/u/228614493?u=45908a4a76165a83ce0b20a474a4d7fd027d67af&v=4" alt="ProPatte" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;"></a>
<img src="https://c8.patreon.com/4/200/35042925/T" alt="That's Ridiculous" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Boris HANSSEN" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Juan Franco" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
<img src="https://ostris.com/wp-content/uploads/2025/08/supporter_default.jpg" alt="Fabrizio Pasqualicchio" width="60" height="60" style="border-radius:8px;margin:5px;display: inline-block;">
</p>

---

