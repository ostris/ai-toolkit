"""Helpers for reporting the captions used by training batches."""

from __future__ import annotations

import json
import os
from contextvars import ContextVar, Token
from typing import Callable, Iterable, Optional


CaptionLogListener = Callable[[str], None]

_caption_log_listener: ContextVar[Optional[CaptionLogListener]] = ContextVar(
    "caption_log_listener",
    default=None,
)


def should_log_captions(completed_step: int, interval: int) -> bool:
    """Return whether a positive, completed step matches the log interval."""
    return completed_step > 0 and interval > 0 and completed_step % interval == 0


def _json_string(value: object) -> str:
    """Render a value as a compact JSON string suitable for a one-line log."""
    if value is not None and not isinstance(value, str):
        value = str(value)
    return json.dumps(value, ensure_ascii=False)


def format_caption_debug_log(
    completed_step: int,
    batches: Iterable[object],
) -> Optional[str]:
    """Format every final caption in the supplied training microbatches."""
    batch_list = list(batches)
    total_microbatches = len(batch_list)
    lines = []

    for microbatch_index, batch in enumerate(batch_list, start=1):
        file_items = getattr(batch, "file_items", None)
        if not file_items:
            continue

        cached_embedding = getattr(batch, "prompt_embeds", None) is not None
        total_items = len(file_items)
        for item_index, item in enumerate(file_items, start=1):
            path = getattr(item, "path", "")
            try:
                basename = os.path.basename(os.fspath(path))
            except TypeError:
                basename = os.path.basename(str(path))

            caption = getattr(item, "caption", None)
            caption_short = getattr(item, "caption_short", None)
            is_reg = bool(getattr(item, "is_reg", False))

            line = (
                f"[caption-debug] step={completed_step} "
                f"microbatch={microbatch_index}/{total_microbatches} "
                f"item={item_index}/{total_items} "
                f"reg={json.dumps(is_reg)} "
                f"cached_embedding={json.dumps(cached_embedding)} "
                f"file={_json_string(basename)} "
                f"caption={_json_string(caption)}"
            )
            if caption_short is not None and caption_short != caption:
                line += f" caption_short={_json_string(caption_short)}"
            lines.append(line)

    return "\n".join(lines) if lines else None


def set_caption_log_listener(listener: Optional[CaptionLogListener]) -> Token:
    """Install a listener in the current context and return its reset token."""
    if listener is not None and not callable(listener):
        raise TypeError("caption log listener must be callable or None")
    return _caption_log_listener.set(listener)


def reset_caption_log_listener(token: Token) -> None:
    """Restore the listener value represented by a previous set token."""
    _caption_log_listener.reset(token)


def emit_caption_log(message: str) -> bool:
    """Send a formatted caption log block to the current context listener."""
    listener = _caption_log_listener.get()
    if listener is None:
        return False
    try:
        listener(message)
    except Exception:
        # A disconnected or faulty debug UI must never interrupt training.
        return False
    return True
