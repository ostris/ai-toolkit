
import os
from pathlib import Path
import subprocess
import torch

class ProcessManager:

    def __init__(self, toolkit_root: str):
        self.toolkit_root = Path(toolkit_root).absolute()

    def validate_gpu(self, gpu_ids: str):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")
        available_gpus = list(range(torch.cuda.device_count()))
        if int(gpu_ids) not in available_gpus:
            raise RuntimeError(f"GPU {gpu_ids} not available")

    def spawn_training_process(self, config_path: str, log_path: str, job_id: str, gpu_ids: str) -> subprocess.Popen:
        env = os.environ.copy()
        env.update({
            'AITK_JOB_ID': job_id,
            'CUDA_VISIBLE_DEVICES': gpu_ids,
            'IS_AI_TOOLKIT_UI': '1'
        })

        python_path = 'python'
        if (self.toolkit_root / '.venv').exists():
            python_path = str(self.toolkit_root / '.venv' / ('Scripts' if os.name == 'nt' else 'bin') / 'python.exe' if os.name == 'nt' else 'python')
        elif (self.toolkit_root / 'venv').exists():
            python_path = str(self.toolkit_root / 'venv' / ('Scripts' if os.name == 'nt' else 'bin') / 'python.exe' if os.name == 'nt' else 'python')

        run_file = self.toolkit_root / 'run.py'
        args = [python_path, str(run_file), config_path, '--log', log_path]

        return subprocess.Popen(
            args,
            env=env,
            cwd=str(self.toolkit_root),
            stdout=open(log_path, 'w'),
            stderr=subprocess.STDOUT,
            text=True
        )
