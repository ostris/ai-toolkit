from tqdm import tqdm


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 1.0)
        kwargs.setdefault("maxinterval", 10.0)
        kwargs.setdefault("miniters", 1)
        kwargs.setdefault("smoothing", 0.05)
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
            super().update(*args, **kwargs)
