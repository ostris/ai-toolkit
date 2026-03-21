# ComfyUI Workflows

These workflows are intended to match the ./config/examples configs one-to-one. So that you can get the most perfect match of what the sampler outputs were. 

The general workflow should be:
1. Drag the .json workflow into ComfyUI
2. Download any missing models and add your trained LoRA into the ComfyUI folders
3. Match the seed and prompt to the training config and sample you want to replicate
4. Run

Note however, even if you get the seed and everything else perfect, the result will still not be exactly the same. There are fundamental differences to how ComfyUI runs the workflows vs how the training is done.