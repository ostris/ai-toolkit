import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# Set ROCm environment variables for better HIP error handling and performance
# These should be set before importing torch
if os.environ.get("AMD_SERIALIZE_KERNEL") is None:
    os.environ["AMD_SERIALIZE_KERNEL"] = "3"  # Better error reporting for HIP errors
if os.environ.get("TORCH_USE_HIP_DSA") is None:
    os.environ["TORCH_USE_HIP_DSA"] = "1"  # Enable device-side assertions
if os.environ.get("HSA_ENABLE_SDMA") is None:
    os.environ["HSA_ENABLE_SDMA"] = "0"  # Disable SDMA for APU compatibility
if os.environ.get("PYTORCH_ROCM_ALLOC_CONF") is None:
    os.environ["PYTORCH_ROCM_ALLOC_CONF"] = "max_split_size_mb:768,garbage_collect=1"  # Better VRAM fragmentation
# HIP_LAUNCH_BLOCKING can be set to "1" for debugging (synchronous kernels), but defaults to "0" for performance
# HSA_OVERRIDE_GFX_VERSION and PYTORCH_ROCM_ARCH should be set by the user or startup script based on their GPU

# Workaround for HIPBLAS errors with quantized models
# ROCBLAS_USE_HIPBLASLT can cause HIPBLAS_STATUS_INTERNAL_ERROR with quantized GEMM operations
# Disable it by default - can be re-enabled via environment variable if needed
if os.environ.get("ROCBLAS_USE_HIPBLASLT") is None:
    os.environ["ROCBLAS_USE_HIPBLASLT"] = "0"  # Disable HIPBLASLT to avoid quantized model crashes
# Reduce ROCBLAS logging overhead
if os.environ.get("ROCBLAS_LOG_LEVEL") is None:
    os.environ["ROCBLAS_LOG_LEVEL"] = "0"  # Disable verbose logging

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            import traceback
            print_acc(f"Error running job: {e}")
            print_acc(f"Traceback: {traceback.format_exc()}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e


if __name__ == '__main__':
    main()
