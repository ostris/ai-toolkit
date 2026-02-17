import os
import threading
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from oxen import Workspace, DataFrame
    OXEN_AVAILABLE = True
except ImportError:
    OXEN_AVAILABLE = False
    Workspace = None
    DataFrame = None

from .oxen_experiment import AIToolkitOxenExperiment

def monitor_train_status(url: str, want_stop_event: threading.Event, finished_event: threading.Event):
    headers = { "Authorization": f"Bearer {os.getenv('OXEN_API_TOKEN')}" }

    while not finished_event.is_set():
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            train_status = response.json()["train_status"]

            if train_status == "stopping":
                want_stop_event.set()
                return
        except Exception as ex:
            print(f"Training status monitor error: {ex}")

        finished_event.wait(timeout=10)

class AIToolkitOxenLogger:
    """
    Logs metrics and saves checkpoints to an Oxen Workspace during AI-toolkit training.
    Adapted from OxenTrainerCallback to work with AI-toolkit's training loop.
    """

    def __init__(
        self, 
        experiment: AIToolkitOxenExperiment, 
        is_main_process: bool = True, 
        fine_tune_id: Optional[str] = None
    ):
        """
        Initializes the logger.

        Args:
            experiment: An initialized AIToolkitOxenExperiment object
            is_main_process: Boolean flag, True if this process should perform Oxen actions
            fine_tune_id: Optional fine-tune ID for workspace naming
        """
        if not OXEN_AVAILABLE:
            print("Warning: Oxen not available, logging will be disabled")
            self.enabled = False
            return
            
        self.experiment = experiment
        self.is_main_process = is_main_process
        self.fine_tune_id = fine_tune_id
        self.enabled = True
        self.log_file_name = "training_logs.jsonl"
        self.log_file_path: Optional[str] = None
        self.workspace: Optional[Workspace] = None
        self.df: Optional[DataFrame] = None
        self.step_count = 0

        if (
            self.is_main_process
            and self.experiment.dir
            and self.experiment.name
            and self.experiment.repo
        ):
            self._initialize_workspace()
            self._start_status_monitor()

    def _start_status_monitor(self):
        base_url = f"{self.experiment.scheme}://{self.experiment.host}"
        repo = self.experiment.repo_id
        self.status_url = f"{base_url}/api/repos/{repo}/fine_tunes/{self.fine_tune_id}/train_status"
        self.want_stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.status_monitor = threading.Thread(
            target=monitor_train_status,
            args=(self.status_url, self.want_stop_event, self.finished_event),
            daemon=True
        )
        self.status_monitor.start()

    def _initialize_workspace(self):
        """Initialize the Oxen workspace and DataFrame."""
        try:
            self.log_file_path = os.path.join(self.experiment.name, self.log_file_name)
            print(f"Oxen Logger: Log file path: {self.log_file_path}")

            print(f"Main process: Initializing Oxen Workspace for branch '{self.experiment.name}'")

            # Ensure experiment directory exists
            print(f"Main process: Ensuring experiment directory exists: {self.experiment.dir}")
            os.makedirs(self.experiment.dir, exist_ok=True)

            # Create initial metrics entry
            initial_metrics = {
                "step": 0,
                "loss": 0.0,
                "learning_rate": 0.0,
                "epoch": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Create pandas DataFrame and save to repo
            initial_df = pd.DataFrame([initial_metrics])
            print(f"Oxen Logger: Initial DataFrame: {initial_df}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            initial_df.to_json(self.log_file_path, orient="records", lines=True)
            print(f"Oxen Logger: Initial DataFrame saved to {self.log_file_path}")

            # Add the initial dataframe to the repo
            try:
                print(f"Oxen Logger: Adding initial DataFrame to repo: {self.log_file_path}")
                self.experiment.repo.add(
                    self.log_file_path, 
                    dst=os.path.dirname(self.log_file_path)
                )
                self.experiment.repo.commit(
                    f"Initial training metrics dataframe: {self.log_file_path}"
                )
                print(f"Main process: ✅ Initial training metrics dataframe added to repo: {self.log_file_path}")
            except Exception as e:
                print(f"Main process: Warning: Could not add initial dataframe to repo: {e}")

            # Initialize workspace and DataFrame
            self.workspace = Workspace(
                self.experiment.repo,
                branch=self.experiment.name,
                workspace_name=self.fine_tune_id,
                path=self.log_file_path,
            )

            self.df = DataFrame(
                self.workspace, 
                self.log_file_path, 
                workspace_name=self.fine_tune_id
            )
            print(f"Main process: DataFrame created successfully")

        except Exception as e:
            print(f"Main process: ERROR Initializing Oxen Workspace/DataFrame: {e}")
            self.workspace = None
            self.df = None
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log training metrics to Oxen.

        Args:
            metrics: Dictionary of metric name -> value
            step: Current training step
        """
        if not self.enabled or not self.is_main_process or self.df is None:
            return

        try:
            # Prepare metrics with step and timestamp
            log_entry = metrics.copy()
            log_entry["step"] = step
            log_entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Insert row into DataFrame
            print(f"Main process: Logging metrics at step {step}: {log_entry}")
            self.df = DataFrame(
                self.workspace,
                self.log_file_path,
                workspace_name=self.fine_tune_id,
            )
            row = self.df.insert_row(log_entry, self.workspace)
            print(f"Main process: Metrics logged successfully")

        except Exception as e:
            print(f"Main process: Error logging metrics to Oxen: {e}")

    def is_stopping(self):
        if not self.is_main_process:
            return False
        # This is set when the fine-tune's train_status was set to "stopping"
        # either from the user hitting "stop" or running out of credits.
        return self.want_stop_event.is_set()

    def get_checkpoint_path(self, step: int) -> str:
        """
        Get the checkpoint path for a given step.

        Args:
            step: Training step number

        Returns:
            Checkpoint path in format: {experiment_name}/checkpoints/step_{N}
        """
        print(f"Main process: Getting checkpoint path for step {step}")
        print(f"Main process: Experiment namesurya-custom: {self.experiment.name}")
        # This is required because the first one is the branch name and the second one is the directory the file is supposed to be saved in
        return f"{self.experiment.name}/checkpoints/step_{step}"

    def save_checkpoint(self, checkpoint_files, step: int):
        """
        Save checkpoint files to Oxen workspace.

        Args:
            checkpoint_files: List of paths to checkpoint files/directories, or a single path string
            step: Current training step
        """
        if not self.enabled or not self.is_main_process or self.workspace is None:
            return

        # Convert single path to list for uniform handling
        if isinstance(checkpoint_files, str):
            checkpoint_files = [checkpoint_files]

        if not checkpoint_files:
            return

        try:
            print(f"Main process: Saving {len(checkpoint_files)} checkpoint file(s) at step {step}")

            # Create a step-specific directory in checkpoints/{finetune_name}/step_{N}
            checkpoint_dst = self.get_checkpoint_path(step)
            print(f"Main process: Checkpoint destination: {checkpoint_dst}")

            for checkpoint_path in checkpoint_files:
                print(f"Main process: Adding checkpoint file: {checkpoint_path}")
                if os.path.isfile(checkpoint_path):
                    # Get just the filename for the destination
                    filename = os.path.basename(checkpoint_path)
                    print(f"Main process: Adding checkpoint file: {checkpoint_path} -> {checkpoint_dst}")
                    self.workspace.add(checkpoint_path, dst=f"{checkpoint_dst}")
                elif os.path.isdir(checkpoint_path):
                    print(f"Main process: Adding checkpoint directory: {checkpoint_path} -> {checkpoint_dst}")
                    # For directories, walk through and add all files
                    dir_name = os.path.basename(checkpoint_path)
                    for root, dirs, files in os.walk(checkpoint_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, checkpoint_path)
                            print(f"Main process: Adding checkpoint file: {file_path}")
                            print(f"relative path: {rel_path}")
                            print(f"directory name: {dir_name}")
                            print(f"Main process: Adding checkpoint file: {file_path} -> {checkpoint_dst}/{dir_name}/{rel_path}")
                            self.workspace.add(file_path, dst=f"{checkpoint_dst}/{dir_name}/{rel_path}")

            print(f"Main process: Checkpoint saved successfully")

        except Exception as e:
            print(f"Main process: Error saving checkpoint to Oxen: {e}")

    def add_samples(self, sample_dir: str, step: int):
        """
        Add sample images to Oxen workspace in the checkpoint's samples directory.

        Args:
            sample_dir: Directory containing sample images
            step: Training step number for organizing samples per checkpoint
        """
        if not self.enabled or not self.is_main_process or self.workspace is None:
            print(f"Main process: Skipping sample images")
            return

        try:
            if not os.path.exists(sample_dir):
                print(f"Main process: Sample directory does not exist: {sample_dir}")
                return

            print(f"Main process: Adding sample images to Oxen workspace from: {sample_dir}")

            # Add all sample images to checkpoints/{finetune_name}/step_{N}/samples/
            checkpoint_path = self.get_checkpoint_path(step)
            dst_path = f"{checkpoint_path}/samples"

            for root, dirs, files in os.walk(sample_dir):
                for file in files:
                    print(f"Main process: Considering sample: {file}")
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4')):
                        file_path = os.path.join(root, file)
                        print(f"Main process: Adding sample image: {file_path} -> {dst_path}")
                        self.workspace.add(file_path, dst=dst_path)

            print(f"Main process: Sample images added to Oxen workspace successfully")

        except Exception as e:
            print(f"Main process: Error adding sample images to Oxen: {e}")

    def finalize_experiment(self, final_model_path: str):
        """
        Finalize the experiment by committing all changes and saving final model.
        
        Args:
            final_model_path: Path to the final trained model
        """
        if not self.enabled or not self.is_main_process or self.workspace is None:
            print(f"Main process: Skipping finalizing experiment")
            return

        try:
            print(f"Main process: Finalizing experiment {self.experiment.name} -> {final_model_path}")
            
            # Save final model if provided
            if final_model_path and os.path.exists(final_model_path):
                print(f"Main process: Saving final model: {final_model_path}")
                
                def add_file_with_rename(file_path, dst_path):
                    """Helper function to add file with potential renaming"""
                    name = self.experiment.name.split("/")[-1]
                    print(f"Main process: Adding file in experiment {name} -> {os.path.basename(file_path)}")
                    if os.path.basename(file_path) == f"{name}.safetensors":
                        # Rename to model.safetensors
                        import shutil
                        temp_dir = os.path.join(os.path.dirname(file_path), "temp_rename")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_file_path = os.path.join(temp_dir, "model.safetensors")
                        shutil.copy2(file_path, temp_file_path)
                        
                        # Add renamed file and cleanup
                        self.workspace.add(temp_file_path, dst=dst_path)
                        shutil.rmtree(temp_dir)
                        print(f"Main process: Saved {temp_file_path} -> {dst_path} (renamed)")
                    else:
                        self.workspace.add(file_path, dst=dst_path)
                        print(f"Main process: Saved {file_path} -> {dst_path}")
                
                if os.path.isfile(final_model_path):
                    add_file_with_rename(final_model_path, self.experiment.name)
                elif os.path.isdir(final_model_path):
                    for root, dirs, files in os.walk(final_model_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, final_model_path)
                            dst_path = os.path.join(self.experiment.name, os.path.dirname(rel_path))
                            add_file_with_rename(file_path, dst_path)

            # Final commit
            self.workspace.commit("Final experiment state with all artifacts")
            print("Main process: Final commit successful")

            self.finished_event.set()

            headers = { "Authorization": f"Bearer {os.getenv('OXEN_API_TOKEN')}" }
            url = f"{self.status_url}/stopped"
            requests.put(url=url, headers=headers, timeout=10)

            self.status_monitor.join(timeout=10)

        except Exception as e:
            print(f"Main process: Error finalizing experiment: {e}")
            raise

    def _format_file_size(self, size_bytes: int) -> str:
        """Convert file size in bytes to human-readable format."""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"