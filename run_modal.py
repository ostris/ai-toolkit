'''

ostris/ai-toolkit on https://modal.com
Run training with the following command:
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml

'''

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import modal
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, "/root/ai-toolkit")
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# define the volume for storing model outputs, using "creating volumes lazily": https://modal.com/docs/guide/volumes
# you will find your model, samples and optimizer stored in: https://modal.com/storage/your-username/main/flux-lora-models
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)

# modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_DIR = "/root/ai-toolkit/modal_output"  # modal_output, due to "cannot mount volume on non-empty path" requirement

# define modal app
image = (
    modal.Image.debian_slim(python_version="3.11")
    # install required system and pip packages, more about this modal approach: https://modal.com/docs/examples/dreambooth_app
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "python-dotenv",
        "torch", 
        "diffusers[torch]", 
        "transformers", 
        "ftfy", 
        "torchvision", 
        "oyaml", 
        "opencv-python", 
        "albumentations",
        "safetensors",
        "lycoris-lora==1.8.3",
        "flatten_json",
        "pyyaml",
        "tensorboard", 
        "kornia", 
        "invisible-watermark", 
        "einops", 
        "accelerate", 
        "toml", 
        "pydantic",
        "omegaconf",
        "k-diffusion",
        "open_clip_torch",
        "timm",
        "prodigyopt",
        "controlnet_aux==0.0.7",
        "bitsandbytes",
        "hf_transfer",
        "lpips", 
        "pytorch_fid", 
        "optimum-quanto", 
        "sentencepiece", 
        "huggingface_hub", 
        "peft"
    )
)

# mount for the entire ai-toolkit directory
# example: "/Users/username/ai-toolkit" is the local directory, "/root/ai-toolkit" is the remote directory
code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")

# create the Modal app with the necessary mounts and volumes
app = modal.App(name="flux-lora-training", image=image, mounts=[code_mount], volumes={MOUNT_DIR: model_volume})

# Check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # Set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)

import argparse
from toolkit.job import get_job

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="A100", # gpu="H100"
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=7200  # 2 hours, increase or decrease if needed
)
def main(config_file_list_str: str, recover: bool = False, name: str = None):
    # convert the config file list from a string to a list
    config_file_list = config_file_list_str.split(",")

    jobs_completed = 0
    jobs_failed = 0

    print(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, name)
            
            job.config['process'][0]['training_folder'] = MOUNT_DIR
            os.makedirs(MOUNT_DIR, exist_ok=True)
            print(f"Training outputs will be saved to: {MOUNT_DIR}")
            
            # run the job
            job.run()
            
            # commit the volume after training
            model_volume.commit()
            
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print(f"Error running job: {e}")
            jobs_failed += 1
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    print_end_message(jobs_completed, jobs_failed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # require at least one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if a job fails
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # optional name replacement for config file
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    args = parser.parse_args()

    # convert list of config files to a comma-separated string for Modal compatibility
    config_file_list_str = ",".join(args.config_file_list)

    main.call(config_file_list_str=config_file_list_str, recover=args.recover, name=args.name)
