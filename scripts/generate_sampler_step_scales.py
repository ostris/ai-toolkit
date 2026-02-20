import argparse
import torch
import os
from diffusers import StableDiffusionPipeline
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add project root to path
sys.path.append(PROJECT_ROOT)

SAMPLER_SCALES_ROOT = os.path.join(PROJECT_ROOT, 'toolkit', 'samplers_scales')


parser = argparse.ArgumentParser(description='Process some images.')
add_arg = parser.add_argument
add_arg('--model', type=str, required=True, help='Path to model')
add_arg('--sampler', type=str, required=True, help='Name of sampler')

args = parser.parse_args()

