
from typing import Callable, Dict, List
from collections import OrderedDict
import threading
import subprocess
from .db_manager import DatabaseManager
from .training_env import TrainingEnvironment
from .process_manager import ProcessManager
from .job_monitor import JobMonitor

class JobManager:

    def __init__(self, toolkit_root: str = ".", db_path: str = "./aitk_db.db"):
        self.db_manager = DatabaseManager(db_path)
        self.training_env = TrainingEnvironment(toolkit_root)
        self.process_manager = ProcessManager(toolkit_root)
        self.job_monitor = JobMonitor(db_path)
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_training(self, job_config: OrderedDict, job_name: str, gpu_ids: str = "0",
                       status_callback: Callable[[Dict], None] = lambda x: None,
                       log_callback: Callable[[str], None] = lambda x: None) -> str:
        try:
            self.process_manager.validate_gpu(gpu_ids)
            job_id = self.db_manager.create_job_entry(job_config, job_name, gpu_ids)
            config_path, log_path, training_folder = self.training_env.setup(job_name, job_config, job_id, self.db_manager.db_path)

            stop_event = threading.Event()
            process = self.process_manager.spawn_training_process(config_path, log_path, job_id, gpu_ids)

            with self._lock:
                self._active_jobs[job_id] = {
                    "process": process,
                    "stop_event": stop_event,
                    "training_folder": training_folder
                }

            status_thread = threading.Thread(
                target=self.job_monitor.monitor_job_status,
                args=(job_id, stop_event, status_callback)
            )
            status_thread.daemon = True
            status_thread.start()

            log_thread = threading.Thread(
                target=self.job_monitor.monitor_logs,
                args=(log_path, stop_event, log_callback)
            )
            log_thread.daemon = True
            log_thread.start()

            completion_thread = threading.Thread(
                target=self._monitor_process_completion,
                args=(process, job_id, stop_event)
            )
            completion_thread.daemon = True
            completion_thread.start()

            return job_id
        except Exception as e:
            status_callback({"job_id": job_id or "unknown", "status": "error", "info": f"Start error: {str(e)}"})
            raise

    def _monitor_process_completion(self, process: subprocess.Popen, job_id: str, stop_event: threading.Event):
        process.wait()
        with self._lock:
            if job_id in self._active_jobs:
                self._active_jobs[job_id]["stop_event"].set()
        self.db_manager.update_job_status_on_failure(job_id, process.returncode)

    def get_job_status(self, job_id: str) -> Dict:
        return self.db_manager.get_job_status(job_id)

    def get_rolling_logs(self, job_id: str, max_lines: int = 100) -> List[str]:
        try:
            with self._lock:
                if job_id not in self._active_jobs:
                    return [f"Error: Job {job_id} not found"]
                log_path = self._active_jobs[job_id]["training_folder"] / "log.txt"
            return self.job_monitor.get_rolling_logs(log_path, max_lines)
        except Exception as e:
            return [f"Log error: {str(e)}"]

    def stop_job(self, job_id: str) -> bool:
        try:
            with self._lock:
                if job_id not in self._active_jobs:
                    return False
                process = self._active_jobs[job_id]["process"]
                stop_event = self._active_jobs[job_id]["stop_event"]

            if self.db_manager.set_stop_flag(job_id):
                process.terminate()
                stop_event.set()
                return True
            return False
        except Exception as e:
            print(f"Stop error for job {job_id}: {str(e)}")
            return False
