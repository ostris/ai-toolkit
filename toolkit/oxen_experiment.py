import os
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from oxen import RemoteRepo
    OXEN_AVAILABLE = True
except ImportError:
    OXEN_AVAILABLE = False
    RemoteRepo = None


class AIToolkitOxenExperiment:
    """
    Manages experiment setup, naming, and branch creation in an Oxen repo for AI-toolkit.
    Adapted from OxenExperiment to work with AI-toolkit's accelerator pattern.
    """

    def __init__(
        self,
        repo_id: str,
        base_model_name: str,
        fine_tuned_model_name: str,
        output_dir_base: str,
        experiment_type: str = "diffusion-training",
        is_main_process: bool = True,
        host: str = "hub.oxen.ai",
        scheme: str = "https",
    ):
        """
        Initializes the experiment.

        Args:
            repo_id: Repository ID in format "namespace/repo_name"
            base_model_name: The base model name (e.g., "black-forest-labs/FLUX.1-dev")
            fine_tuned_model_name: The fine-tuned model name (e.g., "black-forest-labs/FLUX.1-dev-lora")
            output_dir_base: The base directory within the repo for saving outputs
            experiment_type: A prefix for the experiment branch name
            is_main_process: Boolean flag, True if this process should perform setup actions
            host: Host for the Oxen repository (default: "hub.oxen.ai")
            scheme: URL scheme for the repository (default: "https")
        """
        if not OXEN_AVAILABLE:
            raise ImportError("oxen package not available. Install with: pip install oxenai")
            
        self.repo_id = repo_id
        self.host = host
        self.scheme = scheme
        self.repo: Optional[RemoteRepo] = None
        self.output_dir_base = output_dir_base
        self.is_main_process = is_main_process
        self.name: Optional[str] = None
        self.dir: Optional[Path] = None
        self.experiment_number = 0

        if self.is_main_process:
            try:
                # Initialize the remote repo
                print(f"Oxen Experiment: Initializing remote repo: {repo_id} on {scheme}://{host}")
                self.repo = RemoteRepo(repo_id, host=host, scheme=scheme)
                print(f"Oxen Experiment: Remote repo initialized")
                
                branches = self.repo.branches()
                print(f"Oxen Experiment: Branches: {len(branches)}")

                # If no existing branch found, create new one
                if not self.name:
                    # Create base name
                    base_name = f"models/{fine_tuned_model_name}"
                    
                    # Check if branch already exists and make it unique
                    self.name = base_name
                    experiment_number = 0
                    
                    # Keep trying with incremented numbers until we find a unique name
                    while any(branch.name == self.name for branch in branches):
                        self.name = f"{base_name}_v{experiment_number}"
                        experiment_number += 1
                    
                    print(f"Rank 0: Creating new branch '{self.name}'")
                    self.repo.create_checkout_branch(self.name)

                self.dir = Path(os.path.join(self.output_dir_base, self.name))
                self.dir.mkdir(parents=True, exist_ok=True)
                print(f"Main process: Ensured experiment directory exists: {self.dir}")

            except OSError as e:
                print(f"Main process: Error creating directory {self.dir}: {e}")
                raise
            except Exception as e:
                print(f"Main process: Warning - Could not create/checkout branch '{self.name}': {e}")
                raise
        else:
            self.name = None
            self.dir = None
            self.experiment_number = 0

    def get_details_for_broadcast(self) -> dict:
        """Returns details needed by other ranks."""
        if not self.is_main_process:
            return {}
        return {
            'name': self.name,
            'dir': str(self.dir) if self.dir else None,
            'experiment_number': self.experiment_number,
            'repo_url': self.repo_url
        }

    def update_from_broadcast(self, details: dict):
        """Updates instance attributes from broadcasted details on non-main ranks."""
        if self.is_main_process:
            return  # Main process already has the details

        self.name = details.get('name')
        dir_str = details.get('dir')
        self.dir = Path(dir_str) if dir_str else None
        self.experiment_number = details.get('experiment_number', 0)
        # Don't initialize repo on non-main processes
        self.repo = None