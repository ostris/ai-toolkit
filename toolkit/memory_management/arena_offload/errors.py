"""Arena lifecycle error classifications."""


class ArenaSetupFatalError(RuntimeError):
    """Arena setup failed after destructive canonical commit."""


class ArenaCleanupError(RuntimeError):
    """One or more best-effort arena cleanup operations failed."""

    def __init__(self, failures):
        self.failures = tuple(failures)
        detail = "; ".join(
            f"{label}: {type(error).__name__}: {error}"
            for label, error in self.failures
        )
        super().__init__(f"arena runtime cleanup failed: {detail}")


def is_fatal_arena_setup(error) -> bool:
    """Preserve fatal classification through exception wrapper layers."""
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ArenaSetupFatalError):
            return True
        pending.extend(
            (getattr(current, "__cause__", None), getattr(current, "__context__", None))
        )
    return False


def recover_allows_next_job(error, recover: bool) -> bool:
    """Whether the configured sequential runner may continue after error."""
    return bool(recover) and not is_fatal_arena_setup(error)
