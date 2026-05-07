# ComfyUI Workflows

These workflows are intended to match the ./config/examples configs one-to-one, so you can get the closest possible match to the sampler outputs.

The general workflow should be:
1. Drag the .json workflow into ComfyUI
2. Download any missing models and add your trained LoRA into the ComfyUI folders
3. Match the seed and prompt to the training config and sample you want to replicate
4. Run

Note, however, that even if you get the seed and everything else perfect, the result will still not be exactly the same. There are fundamental differences in how ComfyUI runs workflows compared to how the training process is executed.