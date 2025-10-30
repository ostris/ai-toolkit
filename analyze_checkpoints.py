#!/usr/bin/env python3
"""
Analyze LoRA checkpoints to identify most promising ones for motion training.
Ranks checkpoints based on weight magnitudes and ratios without needing ComfyUI testing.
"""

import json
import re
from pathlib import Path
from safetensors import safe_open
import numpy as np
from collections import defaultdict
import torch

def load_metrics(metrics_file):
    """Load metrics.jsonl and return dict keyed by step."""
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            step = data['step']
            metrics[step] = data
    return metrics

def analyze_lora_file(lora_path):
    """
    Analyze a single LoRA safetensors file.
    Returns array of all weights.
    """
    weights = []

    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert to float32 for analysis (handles bfloat16)
            w = tensor.float().cpu().numpy().flatten()
            weights.extend(w)

    return np.array(weights)

def analyze_checkpoint_pair(high_noise_path, low_noise_path):
    """
    Analyze a pair of high_noise and low_noise LoRA files.
    Returns dict with statistics for both.
    """
    high_noise_weights = analyze_lora_file(high_noise_path)
    low_noise_weights = analyze_lora_file(low_noise_path)

    stats = {
        'high_noise': {
            'mean_abs': float(np.mean(np.abs(high_noise_weights))),
            'std': float(np.std(high_noise_weights)),
            'max_abs': float(np.max(np.abs(high_noise_weights))),
            'count': len(high_noise_weights)
        },
        'low_noise': {
            'mean_abs': float(np.mean(np.abs(low_noise_weights))),
            'std': float(np.std(low_noise_weights)),
            'max_abs': float(np.max(np.abs(low_noise_weights))),
            'count': len(low_noise_weights)
        }
    }

    # Calculate ratio
    if stats['low_noise']['mean_abs'] > 0:
        stats['weight_ratio'] = stats['high_noise']['mean_abs'] / stats['low_noise']['mean_abs']
    else:
        stats['weight_ratio'] = float('inf')

    return stats

def score_checkpoint(stats, metrics_at_step):
    """
    Score a checkpoint based on multiple criteria.
    Higher score = more promising for motion LoRA.

    Scoring criteria:
    1. High noise weight magnitude (target: 0.008-0.010)
    2. Weight ratio high/low (target: >1.5x)
    3. Not diverged (loss not too high)
    4. Gradient stability (indicates training health)
    """
    score = 0
    reasons = []

    high_mean = stats['high_noise']['mean_abs']
    low_mean = stats['low_noise']['mean_abs']
    ratio = stats['weight_ratio']

    # Score high noise magnitude (0.008-0.010 is target)
    if 0.008 <= high_mean <= 0.012:
        score += 100
        reasons.append(f"✓ High noise in target range ({high_mean:.6f})")
    elif 0.006 <= high_mean < 0.008:
        score += 60
        reasons.append(f"⚠ High noise slightly low ({high_mean:.6f})")
    elif 0.004 <= high_mean < 0.006:
        score += 30
        reasons.append(f"⚠ High noise weak ({high_mean:.6f})")
    else:
        score += 10
        reasons.append(f"✗ High noise very weak ({high_mean:.6f})")

    # Score weight ratio (>1.5x is target for motion dominance)
    if ratio >= 1.8:
        score += 50
        reasons.append(f"✓ Strong ratio ({ratio:.2f}x)")
    elif ratio >= 1.5:
        score += 35
        reasons.append(f"✓ Good ratio ({ratio:.2f}x)")
    elif ratio >= 1.2:
        score += 20
        reasons.append(f"⚠ Weak ratio ({ratio:.2f}x)")
    else:
        score += 5
        reasons.append(f"✗ Very weak ratio ({ratio:.2f}x)")

    # Penalize if low noise too weak (needs some refinement)
    if low_mean < 0.003:
        score -= 20
        reasons.append(f"⚠ Low noise undertrained ({low_mean:.6f})")
    elif 0.004 <= low_mean <= 0.007:
        score += 20
        reasons.append(f"✓ Low noise good range ({low_mean:.6f})")

    # Consider metrics if available
    if metrics_at_step:
        loss = metrics_at_step.get('loss', 0)
        grad_stab = metrics_at_step.get('gradient_stability', 0)

        # Penalize very high loss (divergence)
        if loss > 0.3:
            score -= 30
            reasons.append(f"✗ High loss ({loss:.4f})")
        elif loss < 0.08:
            score += 10
            reasons.append(f"✓ Low loss ({loss:.4f})")

        # Reward good gradient stability
        if grad_stab > 0.6:
            score += 15
            reasons.append(f"✓ Stable gradients ({grad_stab:.3f})")
        elif grad_stab < 0.4:
            score -= 10
            reasons.append(f"⚠ Unstable gradients ({grad_stab:.3f})")

    return score, reasons

def analyze_training_run(output_dir, run_name):
    """Analyze all checkpoints from a training run."""
    run_dir = Path(output_dir) / run_name
    metrics_file = run_dir / f"metrics_{run_name}.jsonl"

    # Load metrics
    metrics = {}
    if metrics_file.exists():
        metrics = load_metrics(metrics_file)
        print(f"Loaded {len(metrics)} metric entries")
    else:
        print(f"Warning: No metrics file found at {metrics_file}")

    # Find all high_noise checkpoint files
    high_noise_files = sorted(run_dir.glob(f"{run_name}_*_high_noise.safetensors"))

    if not high_noise_files:
        print(f"No checkpoint files found in {run_dir}")
        return

    print(f"Found {len(high_noise_files)} checkpoint pairs\n")
    print("Analyzing checkpoints...")
    print("=" * 100)

    results = []

    for high_noise_path in high_noise_files:
        # Extract step number from filename
        match = re.search(r'_(\d{9})_high_noise', high_noise_path.name)
        if not match:
            continue

        step = int(match.group(1))

        # Find corresponding low_noise file
        low_noise_path = run_dir / f"{run_name}_{match.group(1)}_low_noise.safetensors"
        if not low_noise_path.exists():
            print(f"Warning: Missing low_noise file for step {step}")
            continue

        # Analyze weights
        try:
            stats = analyze_checkpoint_pair(high_noise_path, low_noise_path)
            metrics_at_step = metrics.get(step)
            score, reasons = score_checkpoint(stats, metrics_at_step)

            results.append({
                'step': step,
                'high_noise_file': high_noise_path.name,
                'low_noise_file': low_noise_path.name,
                'stats': stats,
                'metrics': metrics_at_step,
                'score': score,
                'reasons': reasons
            })
            print(f"✓ Step {step}")
        except Exception as e:
            print(f"✗ Error analyzing step {step}: {e}")
            continue

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Print top checkpoints
    print("\nTOP 10 MOST PROMISING CHECKPOINTS:")
    print("=" * 100)

    for i, result in enumerate(results[:10], 1):
        step = result['step']
        score = result['score']
        stats = result['stats']
        metrics = result['metrics']
        reasons = result['reasons']

        print(f"\n#{i} - Step {step} (Score: {score})")
        print(f"    Files: {result['high_noise_file']}")
        print(f"           {result['low_noise_file']}")
        print(f"    High Noise: {stats['high_noise']['mean_abs']:.6f} (±{stats['high_noise']['std']:.6f})")
        print(f"    Low Noise:  {stats['low_noise']['mean_abs']:.6f} (±{stats['low_noise']['std']:.6f})")
        print(f"    Ratio:      {stats['weight_ratio']:.3f}x")

        if metrics:
            print(f"    Loss:       {metrics.get('loss', 'N/A'):.6f}")
            print(f"    LR High:    {metrics.get('lr_0', 'N/A'):.2e}")
            print(f"    LR Low:     {metrics.get('lr_1', 'N/A'):.2e}")
            print(f"    Grad Stab:  {metrics.get('gradient_stability', 'N/A'):.4f}")

        print("    Reasons:")
        for reason in reasons:
            print(f"      {reason}")

    # Print summary statistics
    print("\n" + "=" * 100)
    print("CHECKPOINT PROGRESSION SUMMARY:")
    print("=" * 100)
    print(f"{'Step':<8} {'HN Weight':<12} {'LN Weight':<12} {'Ratio':<8} {'Score':<8} {'Loss':<10}")
    print("-" * 100)

    for result in sorted(results, key=lambda x: x['step']):
        step = result['step']
        hn = result['stats']['high_noise']['mean_abs']
        ln = result['stats']['low_noise']['mean_abs']
        ratio = result['stats']['weight_ratio']
        score = result['score']
        loss = result['metrics'].get('loss', 0) if result['metrics'] else 0

        print(f"{step:<8} {hn:<12.6f} {ln:<12.6f} {ratio:<8.3f} {score:<8} {loss:<10.6f}")

    # Export detailed results to JSON
    output_file = run_dir / f"checkpoint_analysis_{run_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results exported to: {output_file}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_checkpoints.py <run_name> [output_dir]")
        print("\nExample: python analyze_checkpoints.py squ1rtv15")
        print("         python analyze_checkpoints.py squ1rtv16 /path/to/output")
        sys.exit(1)

    run_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/alexis/ai-toolkit/output"

    analyze_training_run(output_dir, run_name)
