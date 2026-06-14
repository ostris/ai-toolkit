import time
from tqdm import tqdm


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # external clock (never affected by tqdm internals)
        self._start = time.perf_counter()
        # store last computed rate
        self._rate = 0.0

    def pause(self):
        pass

    def unpause(self):
        pass

    def update(self, n=1):
        super().update(n)

        now = time.perf_counter()
        elapsed = now - self._start

        if elapsed > 0:
            instant_rate = self.n / elapsed
            # soft EMA smoothing (low lag, no oscillation)
            alpha = 0.05  # lower = smoother, higher = more reactive
            self._rate = (alpha * instant_rate) + ((1 - alpha) * self._rate)

        postfix = getattr(self, "postfix", None)
        if isinstance(postfix, dict):
            new_postfix = dict(postfix)
        else:
            new_postfix = {}

        new_postfix["it/s"] = f"{self._rate:.3f}"
        super().set_postfix(new_postfix, refresh=False)
