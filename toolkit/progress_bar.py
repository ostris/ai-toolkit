from tqdm import tqdm

class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 0)
        kwargs.setdefault("maxinterval", 10.0)
        kwargs.setdefault("miniters", 1)
        kwargs.setdefault("smoothing", 0.05)
        self._instant_rate = None
        self._step_t0 = None
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()
    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()
    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t
    def update(self, *args, **kwargs):
        if not self.paused:
            now = self._time()
            if self._step_t0 is not None and now > self._step_t0:
                self._instant_rate = 1.0 / (now - self._step_t0)
            self._step_t0 = now
            super().update(*args, **kwargs)
    @property
    def format_dict(self):
        d = super().format_dict
        if self._instant_rate is not None:
            d["rate"] = self._instant_rate
        return d
