import sys
import os
from toolkit.accelerator import get_accelerator


def print_acc(*args, sep=' ', end='\n', file=None, flush=False, **kwargs):
    """Print messages in a way that is safe for tqdm progress bars.
    - Only prints on local main process (keeps Accelerate behavior).
    - Uses `tqdm.write` to avoid clobbering progress bars when printing to the console.
    - If `sys.stdout` has been wrapped by `Logger`, let tqdm write to it (it will handle logging).
    - If an explicit `file` is provided (and it's not `sys.stdout`), honor it.
    """
    if not get_accelerator().is_local_main_process:
        return

    # Build the message string like built-in print
    msg = sep.join([str(a) for a in args])

    term = sys.stdout
    is_logger = hasattr(term, 'log') and hasattr(term, 'terminal')

    # If an explicit file is provided and it's not sys.stdout, print directly to it
    if file is not None and file is not term:
        try:
            print(msg, end=end, file=file, flush=flush)
        except Exception:
            # Best-effort fallback
            try:
                print(msg, end=end, flush=flush)
            except Exception:
                pass
        # If we have a separate log file handle, write to it as well (avoid double-write for Logger wrapper)
        if hasattr(term, 'log') and not is_logger:
            try:
                term.log.write(msg + end)
                term.log.flush()
            except Exception:
                pass
        return

    # Default: use tqdm.write to avoid breaking progress bars
    try:
        from tqdm import tqdm
        # tqdm.write writes a newline automatically; direct it to the current stdout wrapper
        tqdm.write(msg, file=term)
    except Exception:
        try:
            # fallback to plain print
            print(msg, end=end, flush=flush)
        except Exception:
            pass

    # If stdout is not the Logger wrapper but does have a .log file, write to it too
    if hasattr(term, 'log') and not is_logger:
        try:
            term.log.write(msg + end)
            term.log.flush()
        except Exception:
            pass


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    sys.stdout = Logger(filename)
    sys.stderr = Logger(filename)
