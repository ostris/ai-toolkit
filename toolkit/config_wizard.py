#!/usr/bin/env python3
"""
Interactive Configuration Wizard for AI Toolkit

Generates optimized training configurations based on your hardware and training goals.
Run with: python -m toolkit.config_wizard
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import psutil


class ConfigWizard:
    """Interactive wizard for generating optimized training configurations."""

    def __init__(self):
        self.answers = {}
        self.config = {}

    def detect_gpu_info(self) -> Tuple[Optional[str], Optional[int]]:
        """Detect GPU model and VRAM using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Take first GPU
                    parts = lines[0].split(',')
                    if len(parts) >= 2:
                        gpu_name = parts[0].strip()
                        vram_mb = int(float(parts[1].strip()))
                        vram_gb = vram_mb // 1024
                        return gpu_name, vram_gb
        except Exception:
            pass
        return None, None

    def detect_ram(self) -> int:
        """Detect system RAM in GB."""
        try:
            ram_bytes = psutil.virtual_memory().total
            ram_gb = ram_bytes // (1024 ** 3)
            return ram_gb
        except Exception:
            return 0

    def detect_storage_type(self, path: str = "/") -> str:
        """Attempt to detect storage type (HDD/SSD/NVMe)."""
        try:
            # Check if running on Linux
            if sys.platform.startswith('linux'):
                result = subprocess.run(
                    ['lsblk', '-d', '-o', 'name,rota'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # rota=1 means rotational (HDD), rota=0 means SSD/NVMe
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] == '0':
                            # Check if NVMe
                            if 'nvme' in parts[0]:
                                return 'nvme'
                            return 'ssd'
                    return 'hdd'
        except Exception:
            pass
        return 'unknown'

    def ask_question(self, question: str, default: Any = None, type_: type = str,
                     choices: Optional[List[str]] = None) -> Any:
        """Ask a question and get user input."""
        if choices:
            print(f"\n{question}")
            for i, choice in enumerate(choices, 1):
                print(f"  {i}. {choice}")
            while True:
                try:
                    if default is not None:
                        prompt = f"Choice [1-{len(choices)}] (default: {default}): "
                    else:
                        prompt = f"Choice [1-{len(choices)}]: "
                    response = input(prompt).strip()
                    if not response and default is not None:
                        return choices[default - 1]
                    idx = int(response)
                    if 1 <= idx <= len(choices):
                        return choices[idx - 1]
                    print(f"Please enter a number between 1 and {len(choices)}")
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input. Please try again.")
        else:
            while True:
                try:
                    if default is not None:
                        prompt = f"\n{question} (default: {default}): "
                    else:
                        prompt = f"\n{question}: "
                    response = input(prompt).strip()
                    if not response and default is not None:
                        return type_(default)
                    if not response:
                        print("This field is required.")
                        continue
                    return type_(response)
                except (ValueError, KeyboardInterrupt) as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    print(f"Invalid input. Please enter a valid {type_.__name__}.")

    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question."""
        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"\n{question} [{default_str}]: ").strip().lower()
            if not response:
                return default
            if response in ['y', 'yes']:
                return True
            if response in ['n', 'no']:
                return False
            print("Please answer 'y' or 'n'")

    def gather_info(self):
        """Gather information from user."""
        print("\n" + "=" * 70)
        print("    AI TOOLKIT - INTERACTIVE CONFIGURATION WIZARD")
        print("=" * 70)
        print("\nThis wizard will help you create an optimized training configuration")
        print("based on your hardware and training goals.\n")

        # GPU detection
        gpu_name, gpu_vram = self.detect_gpu_info()
        if gpu_name and gpu_vram:
            print(f"\n✓ Detected GPU: {gpu_name} ({gpu_vram} GB VRAM)")
            if not self.ask_yes_no("Is this correct?", default=True):
                gpu_vram = self.ask_question("Enter GPU VRAM in GB", type_=int)
        else:
            print("\n✗ Could not detect GPU")
            gpu_vram = self.ask_question("Enter GPU VRAM in GB", type_=int)
        self.answers['gpu_vram'] = gpu_vram

        # RAM detection
        ram_gb = self.detect_ram()
        if ram_gb > 0:
            print(f"\n✓ Detected System RAM: {ram_gb} GB")
            if not self.ask_yes_no("Is this correct?", default=True):
                ram_gb = self.ask_question("Enter system RAM in GB", type_=int)
        else:
            ram_gb = self.ask_question("Enter system RAM in GB", type_=int)
        self.answers['ram_gb'] = ram_gb

        # Storage type
        storage = self.detect_storage_type()
        if storage != 'unknown':
            print(f"\n✓ Detected storage type: {storage.upper()}")
            if not self.ask_yes_no("Is this correct?", default=True):
                storage = self.ask_question(
                    "Select storage type",
                    choices=['HDD', 'SSD', 'NVMe'],
                    default=2
                )
        else:
            storage = self.ask_question(
                "Select storage type",
                choices=['HDD', 'SSD', 'NVMe'],
                default=2
            )
        self.answers['storage'] = storage.lower()

        # Dataset info
        dataset_size = self.ask_question(
            "How many images in your dataset?",
            default=1000,
            type_=int
        )
        self.answers['dataset_size'] = dataset_size

        resolution = self.ask_question(
            "Typical image resolution (e.g., 512, 768, 1024)?",
            default=1024,
            type_=int
        )
        self.answers['resolution'] = resolution

        # Training goal
        training_goal = self.ask_question(
            "What is your training goal?",
            choices=[
                'LoRA (Low-Rank Adaptation)',
                'Full Fine-tuning',
                'DreamBooth',
                'Textual Inversion',
                'Other/Custom'
            ],
            default=1
        )
        self.answers['training_goal'] = training_goal

        # Number of epochs
        epochs = self.ask_question(
            "How many epochs do you plan to train?",
            default=10,
            type_=int
        )
        self.answers['epochs'] = epochs

        # Optimization level
        optimization_level = self.ask_question(
            "Optimization level",
            choices=[
                'Conservative (safe defaults)',
                'Balanced (recommended)',
                'Aggressive (maximum performance)'
            ],
            default=2
        )
        self.answers['optimization_level'] = optimization_level

        # Output path
        output_path = self.ask_question(
            "Where to save the generated config?",
            default="config/optimized.yaml"
        )
        self.answers['output_path'] = output_path

    def calculate_optimal_config(self):
        """Calculate optimal configuration based on gathered info."""
        config = {}

        # Basic job settings
        config['job'] = 'extension'
        config['config'] = {
            'name': 'optimized_training',
            'process': [
                {
                    'type': 'sd_trainer',
                    'training_folder': 'output'
                }
            ]
        }

        # Model configuration (placeholder)
        config['config']['process'][0]['network'] = {
            'type': 'lora',
            'linear': 16,
            'linear_alpha': 16
        }

        # Training configuration
        train_config = {}

        # Calculate batch size based on VRAM and resolution
        vram = self.answers['gpu_vram']
        res = self.answers['resolution']

        # Rough heuristic: VRAM needed = resolution^2 * batch_size * constant
        # For 1024x1024 on 24GB, batch_size ~4-8
        if res <= 512:
            base_batch = min(16, max(4, vram // 3))
        elif res <= 768:
            base_batch = min(8, max(2, vram // 4))
        else:  # 1024+
            base_batch = min(4, max(1, vram // 6))

        train_config['batch_size'] = base_batch

        # Steps and epochs
        train_config['steps'] = self.answers['dataset_size'] * self.answers['epochs'] // base_batch
        train_config['gradient_accumulation_steps'] = 1

        # Learning rate (depends on training goal)
        if 'LoRA' in self.answers['training_goal']:
            train_config['lr'] = 1e-4
        elif 'DreamBooth' in self.answers['training_goal']:
            train_config['lr'] = 5e-6
        elif 'Textual Inversion' in self.answers['training_goal']:
            train_config['lr'] = 5e-3
        else:
            train_config['lr'] = 1e-5

        # Optimizer
        train_config['optimizer'] = 'adamw8bit'

        # Dataloader configuration
        dataset_config = {
            'folder_path': '/path/to/your/dataset',
            'caption_ext': 'txt',
            'resolution': self.answers['resolution']
        }

        # Calculate optimal num_workers based on RAM, dataset size, and optimization level
        ram_gb = self.answers['ram_gb']
        dataset_size = self.answers['dataset_size']
        opt_level = self.answers['optimization_level']

        # Conservative: fewer workers to save memory
        # Balanced: moderate workers
        # Aggressive: more workers for speed
        if 'Conservative' in opt_level:
            worker_factor = 0.5
        elif 'Aggressive' in opt_level:
            worker_factor = 1.5
        else:
            worker_factor = 1.0

        # Heuristic: 1 worker per 8GB RAM, capped at CPU count
        max_workers_by_ram = max(1, int((ram_gb // 8) * worker_factor))
        max_workers_by_cpu = os.cpu_count() or 4
        num_workers = min(max_workers_by_ram, max_workers_by_cpu, 8)

        dataset_config['num_workers'] = num_workers

        # Persistent workers for multi-epoch training
        if self.answers['epochs'] > 1:
            dataset_config['persistent_workers'] = True

        # Caching strategy
        # Calculate cache size estimate
        # Rough: 1024x1024 latent ~4MB, embedding ~2MB per image
        cache_size_mb = dataset_size * 6  # Rough estimate
        cache_size_gb = cache_size_mb / 1024

        # If cache fits comfortably in RAM, use memory cache
        # Otherwise use disk cache
        available_ram_for_cache = ram_gb - (num_workers * 2) - 8  # Reserve for OS and workers

        if cache_size_gb < available_ram_for_cache * 0.7:
            # Memory cache with sharing
            dataset_config['cache_latents'] = True
            dataset_config['cache_latents_to_disk'] = False
            print(f"\n→ Using in-memory cache (~{cache_size_gb:.1f}GB, fits in RAM)")
        else:
            # Disk cache with mmap
            dataset_config['cache_latents'] = False
            dataset_config['cache_latents_to_disk'] = True
            print(f"\n→ Using disk cache (~{cache_size_gb:.1f}GB, exceeds available RAM)")

        # GPU prefetching based on storage type and optimization level
        storage = self.answers['storage']
        if 'Aggressive' in opt_level or storage == 'hdd':
            # Aggressive prefetching for slow storage or aggressive optimization
            dataset_config['gpu_prefetch_batches'] = 3 if storage == 'hdd' else 2
        elif 'Balanced' in opt_level and storage in ['ssd', 'nvme']:
            dataset_config['gpu_prefetch_batches'] = 1
        else:
            # Conservative: no prefetching
            dataset_config['gpu_prefetch_batches'] = 0

        # Batch size auto-scaling for aggressive optimization
        if 'Aggressive' in opt_level:
            train_config['auto_scale_batch_size'] = True
            train_config['min_batch_size'] = max(1, base_batch // 2)
            train_config['max_batch_size'] = base_batch * 2
            train_config['batch_size_warmup_steps'] = 100

        # Mixed precision
        train_config['dtype'] = 'bf16' if vram >= 16 else 'fp16'

        # Gradient checkpointing for memory savings
        if vram < 16:
            train_config['gradient_checkpointing'] = True

        # Save configuration
        config['config']['process'][0]['train'] = train_config
        config['config']['process'][0]['datasets'] = [dataset_config]

        self.config = config

    def generate_yaml(self) -> str:
        """Generate YAML configuration with explanatory comments."""
        lines = []

        # Header
        lines.append("# AI Toolkit - Optimized Training Configuration")
        lines.append("# Generated by Interactive Config Wizard")
        lines.append("#")
        lines.append("# Hardware Profile:")
        lines.append(f"#   GPU VRAM: {self.answers['gpu_vram']} GB")
        lines.append(f"#   System RAM: {self.answers['ram_gb']} GB")
        lines.append(f"#   Storage: {self.answers['storage'].upper()}")
        lines.append("#")
        lines.append("# Training Profile:")
        lines.append(f"#   Dataset size: {self.answers['dataset_size']} images")
        lines.append(f"#   Resolution: {self.answers['resolution']}x{self.answers['resolution']}")
        lines.append(f"#   Goal: {self.answers['training_goal']}")
        lines.append(f"#   Epochs: {self.answers['epochs']}")
        lines.append(f"#   Optimization: {self.answers['optimization_level']}")
        lines.append("")

        # Job and config
        lines.append(f"job: {self.config['job']}")
        lines.append("config:")
        lines.append(f"  name: {self.config['config']['name']}")
        lines.append("  process:")

        process = self.config['config']['process'][0]
        lines.append(f"    - type: {process['type']}")
        lines.append(f"      training_folder: {process['training_folder']}")
        lines.append("")

        # Network config
        lines.append("      # Network architecture")
        lines.append("      network:")
        for key, value in process['network'].items():
            lines.append(f"        {key}: {value}")
        lines.append("")

        # Training config
        lines.append("      # Training parameters")
        lines.append("      train:")
        train = process['train']

        lines.append(f"        # Batch size optimized for {self.answers['gpu_vram']}GB VRAM")
        lines.append(f"        batch_size: {train['batch_size']}")

        if 'auto_scale_batch_size' in train:
            lines.append("")
            lines.append("        # Smart batch size scaling - automatically adjusts for optimal GPU usage")
            lines.append(f"        auto_scale_batch_size: {str(train['auto_scale_batch_size']).lower()}")
            lines.append(f"        min_batch_size: {train['min_batch_size']}")
            lines.append(f"        max_batch_size: {train['max_batch_size']}")
            lines.append(f"        batch_size_warmup_steps: {train['batch_size_warmup_steps']}")

        lines.append("")
        lines.append(f"        steps: {train['steps']}")
        lines.append(f"        gradient_accumulation_steps: {train['gradient_accumulation_steps']}")
        lines.append(f"        lr: {train['lr']}")
        lines.append(f"        optimizer: {train['optimizer']}")
        lines.append(f"        dtype: {train['dtype']}")

        if 'gradient_checkpointing' in train:
            lines.append("")
            lines.append("        # Gradient checkpointing - reduces VRAM usage at cost of speed")
            lines.append(f"        gradient_checkpointing: {str(train['gradient_checkpointing']).lower()}")

        lines.append("")

        # Dataset config
        lines.append("      # Dataset configuration")
        lines.append("      datasets:")
        dataset = process['datasets'][0]

        lines.append(f"        - folder_path: '{dataset['folder_path']}'")
        lines.append(f"          caption_ext: {dataset['caption_ext']}")
        lines.append(f"          resolution: {dataset['resolution']}")
        lines.append("")

        lines.append(f"          # DataLoader workers - {dataset['num_workers']} workers optimized for {self.answers['ram_gb']}GB RAM")
        lines.append(f"          num_workers: {dataset['num_workers']}")

        if 'persistent_workers' in dataset:
            lines.append(f"          # Keep workers alive between epochs - saves {dataset['num_workers']*2}-5+ seconds per epoch")
            lines.append(f"          persistent_workers: {str(dataset['persistent_workers']).lower()}")

        lines.append("")

        # Caching
        if dataset['cache_latents']:
            lines.append("          # In-memory latent caching - fastest but uses RAM")
            lines.append("          # Workers share cached data via shared memory (TODO #2)")
        else:
            lines.append("          # Disk-based latent caching with memory-mapping (TODO #3)")
            lines.append("          # Minimal RAM usage, data loaded on-demand")

        lines.append(f"          cache_latents: {str(dataset['cache_latents']).lower()}")
        lines.append(f"          cache_latents_to_disk: {str(dataset['cache_latents_to_disk']).lower()}")

        if dataset['gpu_prefetch_batches'] > 0:
            lines.append("")
            lines.append(f"          # GPU prefetching - async load next {dataset['gpu_prefetch_batches']} batch(es) to GPU")
            lines.append(f"          # Reduces GPU idle time, especially beneficial for {self.answers['storage'].upper()} storage")
            lines.append(f"          gpu_prefetch_batches: {dataset['gpu_prefetch_batches']}")

        return '\n'.join(lines)

    def save_config(self):
        """Save generated configuration to file."""
        output_path = Path(self.answers['output_path'])

        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate YAML
        yaml_content = self.generate_yaml()

        # Save to file
        with open(output_path, 'w') as f:
            f.write(yaml_content)

        print(f"\n✓ Configuration saved to: {output_path}")
        print(f"\n{'=' * 70}")
        print("CONFIGURATION SUMMARY")
        print('=' * 70)

        # Print summary
        process = self.config['config']['process'][0]
        train = process['train']
        dataset = process['datasets'][0]

        print(f"\nBatch Size: {train['batch_size']}")
        if 'auto_scale_batch_size' in train:
            print(f"  → Auto-scaling enabled: {train['min_batch_size']}-{train['max_batch_size']}")

        print(f"\nDataLoader Workers: {dataset['num_workers']}")
        if dataset.get('persistent_workers'):
            print(f"  → Persistent workers enabled")

        print(f"\nCaching Strategy: ", end='')
        if dataset['cache_latents']:
            print("In-memory (shared)")
        else:
            print("Disk-based (memory-mapped)")

        if dataset['gpu_prefetch_batches'] > 0:
            print(f"\nGPU Prefetching: {dataset['gpu_prefetch_batches']} batches")

        print(f"\nLearning Rate: {train['lr']}")
        print(f"Training Steps: {train['steps']}")
        print(f"Mixed Precision: {train['dtype']}")

        print(f"\n{'=' * 70}")
        print("\nNext steps:")
        print(f"  1. Review and customize: {output_path}")
        print(f"  2. Update 'folder_path' to point to your dataset")
        print(f"  3. Adjust network settings for your model")
        print(f"  4. Run training: python run.py {output_path}")
        print()

    def run(self):
        """Run the wizard."""
        try:
            self.gather_info()
            print("\n" + "=" * 70)
            print("Calculating optimal configuration...")
            self.calculate_optimal_config()
            self.save_config()
            return 0
        except KeyboardInterrupt:
            print("\n\nWizard cancelled by user.")
            return 1
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main entry point."""
    wizard = ConfigWizard()
    return wizard.run()


if __name__ == '__main__':
    sys.exit(main())
