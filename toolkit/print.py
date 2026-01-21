import sys
import os
from toolkit.accelerator import get_accelerator


def print_acc(*args, sep=' ', end='\n', file=None, flush=False, **kwargs):
    """Print messages in a way that is safe for tqdm progress bars *and* ensures
    messages are atomically appended to the log file when `setup_log_to_file` is used.

    Behavior:
    - Still only prints on the local main process (keeps Accelerate behavior).
    - Writes to the terminal using `tqdm.write` (or fallback) so progress bars are not broken.
    - If `sys.stdout` has been wrapped by `Logger`, the message is written to the original
      terminal stream and an atomic append (os.write) is used to append to the log file.
    - If an explicit `file` is provided (and it's not `sys.stdout`) we print to that file
      and also append the message to the log file if present so it stays captured.
    """
    if not get_accelerator().is_local_main_process:
        return

    # Build message and ensure it ends with the provided end
    msg = sep.join([str(a) for a in args])
    if not msg.endswith(end):
        msg = msg + end

    # Detect if sys.stdout is the Logger wrapper
    outer_stdout = sys.stdout
    is_logger = hasattr(outer_stdout, 'log') and hasattr(outer_stdout, 'terminal')

    # We will write to the original terminal stream (if wrapped) so that we can
    # control whether the Logger.log gets written by us atomically.
    terminal_stream = outer_stdout.terminal if is_logger else outer_stdout

    # If an explicit file object is provided (and it's not the stdout wrapper): honor it
    if file is not None and file is not outer_stdout:
        try:
            print(msg, end='', file=file, flush=flush)
        except Exception:
            try:
                print(msg, end='', flush=flush)
            except Exception:
                pass

        # Also append to the log file (if present) using atomic os.write
        if hasattr(outer_stdout, 'log'):
            try:
                fd = outer_stdout.log.fileno()
                os.write(fd, msg.encode('utf-8'))
                if flush:
                    try:
                        os.fsync(fd)
                    except Exception:
                        pass
            except Exception:
                # fallback to normal python write
                try:
                    outer_stdout.log.write(msg)
                    outer_stdout.log.flush()
                except Exception:
                    pass
        return

    # Default: print to terminal safely using tqdm.write
    try:
        from tqdm import tqdm
        # tqdm.write will append a newline; strip our trailing newline to avoid double blank lines
        tqdm.write(msg.rstrip('\n'), file=terminal_stream)
    except Exception:
        try:
            terminal_stream.write(msg)
            terminal_stream.flush()
        except Exception:
            pass

    # If the stdout wrapper has a log file, append atomically to it
    if hasattr(outer_stdout, 'log'):
        try:
            fd = outer_stdout.log.fileno()
            os.write(fd, msg.encode('utf-8'))
            # Only force an fsync when flush requested (keeps perf reasonable)
            if flush:
                try:
                    os.fsync(fd)
                except Exception:
                    pass
        except Exception:
            # fallback to text write which is what Logger.write used to do
            try:
                outer_stdout.log.write(msg)
                outer_stdout.log.flush()
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
