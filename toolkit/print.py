import sys
import os
import time
from toolkit.accelerator import get_accelerator


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        print(*args, **kwargs)


# Progress bars (tqdm etc.) refresh many times a second with \r / cursor-up
# rewrites. The terminal gets every refresh untouched, but writing them all to
# the log file makes it enormous, so transient refreshes are buffered and
# written at most once per interval — each new refresh of the same line(s)
# replaces the buffered one. Real content (anything with actual text and a
# newline) always writes through immediately, preceded by any buffered refresh
# to preserve stream order.
TRANSIENT_WRITE_INTERVAL = 1.0
# Safety valve: an unterminated refresh stream writes through past this size.
MAX_TRANSIENT_BUFFER = 65536


class Logger:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = log_file
        # Last completed refresh cycle (a new cycle replaces the previous one).
        self._cycle = ''
        # In-progress refresh: a lone \r line rewrite, or a multi-write
        # cursor-movement cycle (nested bars) still being assembled.
        self._tail = ''
        self._tail_is_lone_refresh = False
        self._last_transient_write = 0.0

    def write(self, message):
        self.terminal.write(message)
        self._write_log(message)

    def _write_log(self, message):
        has_move = '\r' in message or '\x1b[A' in message
        has_newline = '\n' in message
        if has_move and not has_newline:
            if message.startswith('\r') and '\x1b[A' not in message and (not self._tail or self._tail_is_lone_refresh):
                # Single-line rewrite (plain tqdm bar) — replaces the previous one.
                self._tail = message
                self._tail_is_lone_refresh = True
            else:
                # Part of a multi-write cursor-movement cycle (nested bars).
                self._tail += message
                self._tail_is_lone_refresh = False
        elif self._tail and has_newline and message.strip('\r\n') == '':
            # Pure newline movement closes a multi-write cycle; the completed
            # cycle replaces the previously buffered one.
            self._cycle = self._tail + message
            self._tail = ''
            self._tail_is_lone_refresh = False
        else:
            # Real content — write any buffered refresh first to keep order.
            self._flush_transient()
            self.log.write(message)
            self.log.flush()
            self._last_transient_write = time.monotonic()
            return
        now = time.monotonic()
        # Only flush between cycles (or on a lone \r rewrite, which leaves the
        # cursor on the same row) — flushing mid-cycle would leave the file's
        # cursor moved up and misalign everything written after.
        mid_cycle = self._tail and not self._tail_is_lone_refresh
        if (now - self._last_transient_write >= TRANSIENT_WRITE_INTERVAL and not mid_cycle) or len(
                self._tail) > MAX_TRANSIENT_BUFFER:
            self._flush_transient()
            self._last_transient_write = now

    def _flush_transient(self):
        if self._cycle or self._tail:
            self.log.write(self._cycle + self._tail)
            self.log.flush()
            self._cycle = ''
            self._tail = ''
            self._tail_is_lone_refresh = False

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    # Capture the real streams before replacing them — wrapping the
    # already-replaced sys.stdout as the stderr Logger's "terminal" would
    # double-write every stderr message to the file. Both wrappers share a
    # single file handle.
    log_file = open(filename, 'a')
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)
