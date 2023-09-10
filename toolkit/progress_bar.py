from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            self.start_t += self._time() - self.last_time

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)
