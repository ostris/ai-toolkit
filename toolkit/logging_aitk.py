from typing import OrderedDict, Optional
from PIL import Image

from toolkit.config_modules import LoggingConfig
import os
import sqlite3
import time
from typing import Any, Dict, Tuple, List


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

    # finish logging
    def finish(self):
        pass


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
        image = self._image(image, caption=caption, *args, **kwargs)
        self._log({f"sample_{id}": image}, commit=False)

    def finish(self):
        self.run.finish()


class UILogger:
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
    if logging_config.use_wandb:
        project_name = logging_config.project_name
        run_name = logging_config.run_name
        return WandbLogger(project=project_name, run_name=run_name, config=all_config)
    elif logging_config.use_ui_logger:
        if save_root is None:
            raise ValueError("save_root must be provided when using UILogger")
        log_file = os.path.join(save_root, "loss_log.db")
        return UILogger(log_file=log_file)
    else:
        return EmptyLogger()
