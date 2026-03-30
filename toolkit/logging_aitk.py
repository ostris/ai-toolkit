from typing import Any, Dict, List, OrderedDict, Optional, Tuple
from PIL import Image

from toolkit.config_modules import LoggingConfig
import os
import random
import re
import sqlite3
import string
import time
import uuid


# Base logger class
# This class does nothing, it's just a placeholder
class EmptyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

    # start logging the training
    def start(self):
        pass

    # collect the log to send
    def log(self, *args, **kwargs):
        pass

    # send the log
    def commit(self, step: Optional[int] = None):
        pass

    # log image
    def log_image(self, *args, **kwargs):
        pass

    # log checkpoint artifact
    def log_checkpoint(self, file_path: str):
        pass

    # register trained model
    def log_model(self, **kwargs):
        pass

    # log training datasets
    def log_datasets(self, dataset_configs):
        pass

    # finish logging
    def finish(self):
        pass


def _make_lora_pyfunc_stub():
    """Build a minimal pyfunc stub so the LoRA can be registered in the Model Registry.

    This is a stopgap until MLflow gets a native diffusers adapter flavor —
    see https://github.com/mlflow/mlflow/issues/22122.
    The stub stores the LoRA weights as an artifact for lineage tracking but
    does not implement inference.  To use the LoRA, load it with diffusers, e.g.:
        pipe.load_lora_weights(context.artifacts['lora_weights'])
    """
    import mlflow.pyfunc

    class LoRAModelStub(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            raise NotImplementedError(
                "This LoRA adapter cannot be served directly via MLflow. "
                "Load the weights from context.artifacts['lora_weights'] "
                "with your preferred diffusers pipeline. "
                "See https://github.com/mlflow/mlflow/issues/22122"
            )

    return LoRAModelStub


# Wandb logger class
# This class logs the data to wandb
class WandbLogger(EmptyLogger):
    def __init__(self, project: str, run_name: str | None, config: OrderedDict) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config

    def start(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "Failed to import wandb. Please install wandb by running `pip install wandb`"
            )

        # send the whole config to wandb
        run = wandb.init(project=self.project, name=self.run_name, config=self.config)
        self.run = run
        self._log = wandb.log  # log function
        self._image = wandb.Image  # image object

    def log(self, *args, **kwargs):
        # when commit is False, wandb increments the step,
        # but we don't want that to happen, so we set commit=False
        self._log(*args, **kwargs, commit=False)

    def commit(self, step: Optional[int] = None):
        # after overall one step is done, we commit the log
        # by log empty object with commit=True
        self._log({}, step=step, commit=True)

    def log_image(
        self,
        image: Image,
        id,  # sample index
        caption: str | None = None,  # positive prompt
        *args,
        **kwargs,
    ):
        # create a wandb image object and log it
        # W&B associates images with the step passed to the next commit() call
        kwargs.pop("step", None)
        image = self._image(image, caption=caption, *args, **kwargs)
        self._log({f"sample_{id}": image}, commit=False)

    def finish(self):
        self.run.finish()


class MLflowLogger(EmptyLogger):
    """MLflow experiment tracking logger.

    Follows the same two-phase pattern as WandbLogger:
    - log() accumulates metrics into a buffer
    - commit(step) flushes the buffer to MLflow as a single batched call

    Uses the MLflow fluent API (mlflow.start_run / mlflow.log_metrics / etc.).
    All MLflow API calls are wrapped in try/except so a tracking server failure
    never kills a training run.
    """

    # MLflow metric keys: alphanumerics, underscores, dashes, periods, spaces, slashes
    _METRIC_KEY_RE = re.compile(r"[^a-zA-Z0-9_/.\- ]+")

    def __init__(
        self,
        project: str,
        run_name: str | None,
        config: OrderedDict,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        log_artifacts: bool = False,
        register_model: bool = False,
        registered_model_name: str | None = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or project
        self.log_artifacts = log_artifacts
        self.register_model = register_model
        self.registered_model_name = registered_model_name

        self._pending: Dict[str, float] = {}
        self._mlflow = None
        self._run = None
        self._started = False
        self._last_step: Optional[int] = None
        self._logged_images_tag_set = False

    def start(self):
        if self._started:
            return

        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "Failed to import mlflow. Please install a compatible version by running `pip install \"mlflow>=3,<4\"`"
            )

        self._mlflow = mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        try:
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name)
        except Exception as e:
            print(f"[MLflowLogger] Failed to start MLflow run: {e}")
            self._mlflow = None
            return

        self._started = True

        # Log flattened config as params for comparison
        flat_params = self._flatten_config(self.config)
        if flat_params:
            try:
                items = list(flat_params.items())
                for i in range(0, len(items), 100):
                    batch = dict(items[i : i + 100])
                    mlflow.log_params(batch)
            except Exception as e:
                print(f"[MLflowLogger] Warning: failed to log params: {e}")

    def log(self, log_dict=None, *args, **kwargs):
        if log_dict is None:
            if args:
                log_dict = args[0]
            else:
                return

        if not isinstance(log_dict, dict):
            return

        for k, v in log_dict.items():
            key = self._sanitize_key(k)
            try:
                self._pending[key] = float(v)
            except (TypeError, ValueError):
                pass

    def commit(self, step: Optional[int] = None):
        if self._mlflow is None:
            return

        self._last_step = step

        if self._pending:
            try:
                self._mlflow.log_metrics(self._pending, step=step)
                self._pending.clear()
            except Exception as e:
                print(f"[MLflowLogger] Warning: failed to log metrics at step {step}: {e}")
                self._pending.clear()

    def log_image(
        self,
        image,
        id,  # sample index
        caption: str | None = None,
        step: Optional[int] = None,
        *args,
        **kwargs,
    ):
        if self._mlflow is None:
            return

        # handle video frames (list of images) — log only the first frame
        if isinstance(image, list):
            if len(image) == 0:
                return
            image = image[0]

        try:
            import numpy as np
            is_loggable = isinstance(image, (Image.Image, np.ndarray))
        except ImportError:
            is_loggable = isinstance(image, Image.Image)
            np = None

        if not is_loggable:
            return

        # Convert numpy to PIL for consistent handling
        if np is not None and isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if step is None:
            step = self._last_step if self._last_step is not None else 0
        try:
            # Step-aligned images for the Image Grid chart in Model Metrics.
            # We build the artifact path manually using '+' as the separator
            # instead of relying on mlflow.log_image(key=, step=) because
            # MLflow <= 3.10.1 uses '%' which breaks URL encoding for certain
            # step numbers (e.g. step 23 → '%23' → '#'). Fixed on master via
            # mlflow/mlflow#21269 but not yet in a stable release.
            # The JS parser (ImageReducer.ts) already supports both '+' and '%'.
            ts = int(time.time() * 1000)
            file_uuid = f"{random.choice(string.ascii_lowercase[6:])}{str(uuid.uuid4())[1:]}"
            safe_key = str(id).replace("/", "#")

            base = f"images/{safe_key}+step+{step}+timestamp+{ts}+{file_uuid}"
            self._mlflow.log_image(image, artifact_file=f"{base}.png")

            if not self._logged_images_tag_set and self._run:
                self._mlflow.set_tag("mlflow.loggedImages", "true")
                self._logged_images_tag_set = True
        except Exception as e:
            print(f"[MLflowLogger] Warning: failed to log image sample_{id}: {e}")

    def finish(self):
        if self._mlflow is None or not self._started:
            return

        if self._pending:
            try:
                self._mlflow.log_metrics(self._pending, step=self._last_step)
            except Exception as e:
                print(f"[MLflowLogger] Warning: failed to flush final metrics: {e}")
            self._pending.clear()

        try:
            self._mlflow.end_run()
        except Exception as e:
            print(f"[MLflowLogger] Warning: failed to end run: {e}")

        self._run = None
        self._started = False
        self._mlflow = None

    @property
    def run_id(self) -> str | None:
        """Return the MLflow run ID."""
        if self._run is not None:
            return self._run.info.run_id
        return None

    def log_checkpoint(self, file_path: str):
        """Log a saved checkpoint as an MLflow artifact."""
        if self._mlflow is None or not self.log_artifacts:
            return
        if not os.path.exists(file_path):
            print(f"[MLflowLogger] Warning: checkpoint path does not exist, skipping: {file_path}")
            return

        try:
            if os.path.isdir(file_path):
                self._mlflow.log_artifacts(file_path, artifact_path="checkpoints")
            else:
                self._mlflow.log_artifact(file_path, artifact_path="checkpoints")
        except Exception as e:
            print(f"[MLflowLogger] Warning: failed to log checkpoint artifact {file_path}: {e}")

    def log_model(
        self,
        lora_path: str,
        base_model: str,
        model_type: str = "sd1",
        network_type: str = "lora",
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        **kwargs,
    ):
        """Register the LoRA adapter in the MLflow Model Registry.

        Uses a minimal pyfunc stub so the LoRA appears in the Models section
        with versioning and lineage.  Inference is not supported — load the
        weights with diffusers directly.
        See https://github.com/mlflow/mlflow/issues/22122.
        """
        if not self.register_model:
            return
        if self._mlflow is None or not self._started:
            return
        if not os.path.exists(lora_path):
            print(f"[MLflowLogger] Warning: LoRA path does not exist, skipping registration: {lora_path}")
            return

        try:
            import mlflow.pyfunc

            LoRAModelStub = _make_lora_pyfunc_stub()

            model_config = {
                "base_model": base_model,
                "model_type": model_type,
                "network_type": network_type,
            }
            if lora_rank is not None:
                model_config["lora_rank"] = lora_rank
            if lora_alpha is not None:
                model_config["lora_alpha"] = lora_alpha

            mlflow.pyfunc.log_model(
                name="lora_model",
                python_model=LoRAModelStub(),
                artifacts={"lora_weights": lora_path},
                model_config=model_config,
                registered_model_name=self.registered_model_name,
            )

            print(f"[MLflowLogger] Registered LoRA model from {lora_path}")
            if self.registered_model_name:
                print(f"[MLflowLogger] Model registered as '{self.registered_model_name}'")
        except Exception as e:
            print(f"[MLflowLogger] Warning: failed to register LoRA model: {e}")

    def log_datasets(self, dataset_configs):
        """Log training datasets to MLflow so they appear in the Datasets section."""
        if self._mlflow is None or not self._started:
            return

        try:
            import pandas as pd

            for i, ds in enumerate(dataset_configs):
                source_path = ds.folder_path or ds.dataset_path or "unknown"
                name = os.path.basename(source_path) if source_path != "unknown" else f"dataset_{i}"

                info = {
                    "source_path": [source_path],
                    "resolution": [str(ds.resolution)],
                    "type": [ds.type],
                }
                if ds.caption_ext:
                    info["caption_ext"] = [ds.caption_ext]
                if ds.caption_dropout_rate is not None:
                    info["caption_dropout_rate"] = [ds.caption_dropout_rate]
                if ds.trigger_word:
                    info["trigger_word"] = [ds.trigger_word]

                df = pd.DataFrame(info)
                dataset = self._mlflow.data.from_pandas(df, name=name, source=source_path)
                self._mlflow.log_input(dataset, context="training")
        except Exception as e:
            print(f"[MLflowLogger] Warning: failed to log datasets: {e}")

    # ---- internal helpers ----

    def _sanitize_key(self, key: str) -> str:
        return self._METRIC_KEY_RE.sub("_", key)

    @staticmethod
    def _flatten_config(config, prefix: str = "", sep: str = ".") -> Dict[str, str]:
        """Flatten a nested dict/OrderedDict into dot-separated keys with string values."""
        flat = {}
        if not isinstance(config, (dict, OrderedDict)):
            return flat

        for k, v in config.items():
            full_key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, (dict, OrderedDict)):
                flat.update(MLflowLogger._flatten_config(v, full_key, sep))
            else:
                flat[full_key] = str(v)[:250]
        return flat


class CompositeLogger(EmptyLogger):
    """Dispatches all logging calls to multiple loggers simultaneously.

    Enables running W&B + MLflow (+ UILogger) at the same time.
    """

    def __init__(self, loggers: list) -> None:
        self._loggers = [lg for lg in loggers if type(lg) is not EmptyLogger]

    def _safe_call(self, method_name, *args, **kwargs):
        for lg in self._loggers:
            try:
                getattr(lg, method_name)(*args, **kwargs)
            except ImportError:
                raise  # missing package = config error, never swallow
            except Exception as e:
                print(f"[CompositeLogger] {type(lg).__name__}.{method_name}() failed: {e}")

    def start(self):
        self._safe_call("start")

    def log(self, *args, **kwargs):
        self._safe_call("log", *args, **kwargs)

    def commit(self, step: Optional[int] = None):
        self._safe_call("commit", step=step)

    def log_image(self, *args, **kwargs):
        self._safe_call("log_image", *args, **kwargs)

    def log_checkpoint(self, file_path: str):
        self._safe_call("log_checkpoint", file_path)

    def log_model(self, **kwargs):
        self._safe_call("log_model", **kwargs)

    def log_datasets(self, dataset_configs):
        self._safe_call("log_datasets", dataset_configs)

    def finish(self):
        self._safe_call("finish")


class UILogger(EmptyLogger):
    def __init__(
        self,
        log_file: str,
        flush_every_n: int = 256,
        flush_every_secs: float = 0.25,
    ) -> None:
        self.log_file = log_file
        self._log_to_commit: Dict[str, Any] = {}

        self._con: Optional[sqlite3.Connection] = None
        self._started = False

        self._step_counter = 0

        # buffered writes
        self._pending_steps: List[Tuple[int, float]] = []
        self._pending_metrics: List[
            Tuple[int, str, Optional[float], Optional[str]]
        ] = []
        self._pending_key_minmax: Dict[str, Tuple[int, int]] = {}

        self._flush_every_n = int(flush_every_n)
        self._flush_every_secs = float(flush_every_secs)
        self._last_flush = time.time()

    # start logging the training
    def start(self):
        if self._started:
            return

        parent = os.path.dirname(os.path.abspath(self.log_file))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        self._con = sqlite3.connect(self.log_file, timeout=30.0, isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL;")
        self._con.execute("PRAGMA synchronous=NORMAL;")
        self._con.execute("PRAGMA temp_store=MEMORY;")
        self._con.execute("PRAGMA foreign_keys=ON;")
        self._con.execute("PRAGMA busy_timeout=30000;")

        self._init_schema(self._con)

        self._started = True
        self._last_flush = time.time()

    # collect the log to send
    def log(self, log_dict):
        # log_dict is like {'learning_rate': learning_rate}
        if not isinstance(log_dict, dict):
            raise TypeError("log_dict must be a dict")
        self._log_to_commit.update(log_dict)

    # send the log
    def commit(self, step: Optional[int] = None):
        if not self._started:
            self.start()

        if not self._log_to_commit:
            return

        if step is None:
            step = self._step_counter
            self._step_counter += 1
        else:
            step = int(step)
            if step >= self._step_counter:
                self._step_counter = step + 1

        wall_time = time.time()

        # buffer step row (upsert later)
        self._pending_steps.append((step, wall_time))

        # buffer metrics rows + key min/max updates
        for k, v in self._log_to_commit.items():
            k = k if isinstance(k, str) else str(k)
            vr, vt = self._coerce_value(v)

            self._pending_metrics.append((step, k, vr, vt))

            if k in self._pending_key_minmax:
                lo, hi = self._pending_key_minmax[k]
                if step < lo:
                    lo = step
                if step > hi:
                    hi = step
                self._pending_key_minmax[k] = (lo, hi)
            else:
                self._pending_key_minmax[k] = (step, step)

        self._log_to_commit = {}

        # flush conditions
        now = time.time()
        if (
            len(self._pending_metrics) >= self._flush_every_n
            or (now - self._last_flush) >= self._flush_every_secs
        ):
            self._flush()

    # log image
    def log_image(self, *args, **kwargs):
        # this doesnt log images for now
        pass


    # finish logging
    def finish(self):
        if not self._started:
            return

        self._flush()

        assert self._con is not None
        self._con.close()
        self._con = None
        self._started = False

    # -------------------------
    # internal
    # -------------------------

    def _init_schema(self, con: sqlite3.Connection) -> None:
        con.execute("BEGIN;")

        con.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step      INTEGER PRIMARY KEY,
                wall_time REAL NOT NULL
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS metric_keys (
                key             TEXT PRIMARY KEY,
                first_seen_step INTEGER,
                last_seen_step  INTEGER
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                step       INTEGER NOT NULL,
                key        TEXT NOT NULL,
                value_real REAL,
                value_text TEXT,
                PRIMARY KEY (step, key),
                FOREIGN KEY (step) REFERENCES steps(step) ON DELETE CASCADE
            );
        """)

        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_key_step ON metrics (key, step);"
        )

        con.execute("COMMIT;")

    def _coerce_value(self, v: Any) -> Tuple[Optional[float], Optional[str]]:
        if v is None:
            return None, None
        if isinstance(v, bool):
            return float(int(v)), None
        if isinstance(v, (int, float)):
            return float(v), None
        try:
            return float(v), None  # type: ignore[arg-type]
        except Exception:
            return None, str(v)

    def _flush(self) -> None:
        if not self._pending_steps and not self._pending_metrics:
            return

        assert self._con is not None
        con = self._con

        con.execute("BEGIN;")

        # steps upsert
        if self._pending_steps:
            con.executemany(
                "INSERT INTO steps(step, wall_time) VALUES(?, ?) "
                "ON CONFLICT(step) DO UPDATE SET wall_time=excluded.wall_time;",
                self._pending_steps,
            )

        # keys table upsert (maintains list of keys + seen range)
        if self._pending_key_minmax:
            con.executemany(
                "INSERT INTO metric_keys(key, first_seen_step, last_seen_step) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "first_seen_step=MIN(metric_keys.first_seen_step, excluded.first_seen_step), "
                "last_seen_step=MAX(metric_keys.last_seen_step, excluded.last_seen_step);",
                [(k, lo, hi) for k, (lo, hi) in self._pending_key_minmax.items()],
            )

        # metrics upsert
        if self._pending_metrics:
            con.executemany(
                "INSERT INTO metrics(step, key, value_real, value_text) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(step, key) DO UPDATE SET "
                "value_real=excluded.value_real, value_text=excluded.value_text;",
                self._pending_metrics,
            )

        con.execute("COMMIT;")

        self._pending_steps.clear()
        self._pending_metrics.clear()
        self._pending_key_minmax.clear()
        self._last_flush = time.time()


# create logger based on the logging config
def create_logger(
    logging_config: LoggingConfig,
    all_config: OrderedDict,
    save_root: Optional[str] = None,
):
    loggers: List[EmptyLogger] = []

    if logging_config.use_wandb:
        loggers.append(
            WandbLogger(
                project=logging_config.project_name,
                run_name=logging_config.run_name,
                config=all_config,
            )
        )

    if logging_config.use_mlflow:
        loggers.append(
            MLflowLogger(
                project=logging_config.project_name,
                run_name=logging_config.run_name,
                config=all_config,
                tracking_uri=logging_config.mlflow_tracking_uri,
                experiment_name=logging_config.mlflow_experiment_name,
                log_artifacts=logging_config.mlflow_log_artifacts,
                register_model=logging_config.mlflow_register_model,
                registered_model_name=logging_config.mlflow_registered_model_name,
            )
        )

    if logging_config.use_ui_logger:
        if save_root is None:
            raise ValueError("save_root must be provided when using UILogger")
        log_file = os.path.join(save_root, "loss_log.db")
        loggers.append(UILogger(log_file=log_file))

    if len(loggers) == 0:
        return EmptyLogger()
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)
