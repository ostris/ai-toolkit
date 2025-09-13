
from pathlib import Path
from collections import OrderedDict
import json

class TrainingEnvironment:

    def __init__(self, toolkit_root: str):
        self.toolkit_root = Path(toolkit_root).absolute()

    def setup(self, job_name: str, job_config: OrderedDict, job_id: str, db_path: str) -> tuple[str, str, str]:
        training_root = self.toolkit_root / "output"
        training_folder = training_root / job_name
        training_folder.mkdir(parents=True, exist_ok=True)

        log_path = training_folder / "log.txt"
        if log_path.exists():
            logs_folder = training_folder / "logs"
            logs_folder.mkdir(exist_ok=True)
            num = 0
            while (logs_folder / f"{num}_log.txt").exists():
                num += 1
            log_path.rename(logs_folder / f"{num}_log.txt")

        config_path = training_folder / ".job_config.json"
        job_config['config']['process'][0]['sqlite_db_path'] = str(db_path)
        with open(config_path, 'w') as f:
            json.dump(job_config, f, indent=2)

        return str(config_path), str(log_path), str(training_folder)
