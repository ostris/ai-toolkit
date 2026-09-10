"""The inference engine as an AI Toolkit job: a resident generation server
that stays up until the job is stopped.

    job: extension
    config:
      name: inference_engine
      process:
        - type: InferenceEngine
          sqlite_db_path: ./aitk_db.db
          device: cuda
          engine:
            job_folder: output/inference_engine   # engine.json + outputs/ + assets/
            default_model: {arch: zimage}         # optional warm load

The server binds 127.0.0.1 on an ephemeral port and publishes it (with a
per-launch token) to <job_folder>/engine.json; the UI proxies to it.
"""

import json
import os
import secrets
import signal
import time
from collections import OrderedDict

import torch

from jobs.process import BaseExtensionProcess
from toolkit.paths import TOOLKIT_ROOT
from toolkit.ui_job_status import UIJobStatus


class InferenceEngineConfig:
    def __init__(self, **kwargs):
        self.job_folder = kwargs.get("job_folder", None)
        self.output_folder = kwargs.get("output_folder", None)
        self.default_model = kwargs.get("default_model", None)
        self.dtype = kwargs.get("dtype", "bf16")
        self.host = kwargs.get("host", "127.0.0.1")
        # 0 = ephemeral (recommended); a fixed port is for external clients
        self.port = int(kwargs.get("port", 0))
        self.token = kwargs.get("token", None)


class InferenceEngine(BaseExtensionProcess):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.ui = UIJobStatus(self.config.get("sqlite_db_path", "./aitk_db.db"))
        self.engine_config = InferenceEngineConfig(**self.get_conf("engine", {}))
        self.device = self.get_conf("device", "cuda")
        job_folder = self.engine_config.job_folder or os.path.join(TOOLKIT_ROOT, "output", self.name)
        self.job_folder = os.path.abspath(job_folder)
        self.output_folder = os.path.abspath(
            self.engine_config.output_folder or os.path.join(self.job_folder, "outputs")
        )
        self.endpoint_path = os.path.join(self.job_folder, "engine.json")
        self.engine = None
        self.server = None
        self.server_thread = None
        self.endpoint = None

    def run(self):
        super().run()
        from .engine import Engine
        from .server import create_app

        self.ui.update_status("running", "Starting server")
        os.makedirs(self.job_folder, exist_ok=True)
        self.engine = Engine(
            device=self.device,
            output_folder=self.output_folder,
            assets_folder=os.path.join(self.job_folder, "assets"),
            status_cb=self.ui.update_info,
            dtype=self.engine_config.dtype,
        )
        token = self.engine_config.token or secrets.token_urlsafe(24)
        self.start_server(create_app(self.engine, token), token)

        def _on_signal(signum, frame):
            # first signal: stop after the current request; if a request (or a
            # model load) is in flight, interrupt it right here on the main
            # thread so a long load cannot hold the shutdown hostage
            self.engine.stop("stopped")
            if self.engine.current is not None or self.engine.loading:
                raise KeyboardInterrupt()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _on_signal)
            except Exception:
                pass

        def _on_db_stop(reason):
            self.engine.stop(reason)
            # deliver a real SIGINT to the main thread (never os.kill: on
            # Windows that is TerminateProcess)
            signal.raise_signal(signal.SIGINT)

        self.ui.start_stop_watcher(_on_db_stop)

        if self.engine_config.default_model:
            with torch.no_grad():
                self.engine.preload(dict(self.engine_config.default_model))

        self.engine._idle_status()
        try:
            with torch.no_grad():
                self.engine.run_forever()
        finally:
            self.shutdown()

    def start_server(self, app, token: str):
        from .server import serve_in_thread

        self.server, self.server_thread, host, port = serve_in_thread(
            app, self.engine_config.host, self.engine_config.port
        )
        self.endpoint = {
            "host": host,
            "port": port,
            "token": token,
            "pid": os.getpid(),
            "job_id": self.ui.job_id,
            "device": self.device,
            "job_folder": self.job_folder,
            "output_folder": self.output_folder,
            "started_at": time.time(),
        }
        tmp = self.endpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.endpoint, f, indent=2)
        os.replace(tmp, self.endpoint_path)
        print(f"[AITK] Inference engine listening on http://{host}:{port} (endpoint file: {self.endpoint_path})")

    def shutdown(self):
        reason = self.engine.stop_reason if self.engine else "stopped"
        print(f"[AITK] Inference engine shutting down ({reason})")
        if self.server is not None:
            self.server.should_exit = True
            if self.server_thread is not None:
                self.server_thread.join(timeout=10)
        try:
            os.remove(self.endpoint_path)
        except FileNotFoundError:
            pass
        if self.engine is not None:
            try:
                self.engine.unload()
            except Exception:
                pass
        self.ui.update_status("stopped", "Engine stopped")

    def on_error(self, e: Exception):
        super().on_error(e)
        if isinstance(e, KeyboardInterrupt):
            self.ui.update_status("stopped", "Engine stopped")
        else:
            self.ui.update_status("error", str(e))
        try:
            os.remove(self.endpoint_path)
        except Exception:
            pass
