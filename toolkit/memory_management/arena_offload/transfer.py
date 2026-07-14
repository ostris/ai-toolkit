"""Compact host-to-device transfer runtime for arena offload.

Owns fetch plans at execution time, the reusable device ring, custom fetch
operators, statistics, and checkpoint/recompute ticket lifetime.
"""

from __future__ import annotations

import collections
import contextlib
import itertools
import threading
import time
from dataclasses import dataclass

import torch
from toolkit.memory_management import pin_manager
from .ownership import validate_process_owner

@dataclass
class _Ticket:
    tid: int
    device_buffer: torch.Tensor
    ready_event: torch.cuda.Event | None
    slot: int = -1
    # The exact key the slot was acquired under: torch.device("cuda") and
    # buffer.device (cuda:0) do not compare equal as dict keys.
    slot_device: torch.device | None = None
    free_event: torch.cuda.Event | None = None
    h2d_start: torch.cuda.Event | None = None
    h2d_end: torch.cuda.Event | None = None
    nbytes: int = 0
    copies: int = 0


class _Slot:
    """One reusable device buffer in the fetch ring.

    The buffer is allocated once and reused for every fetch that lands on this
    slot. ``free_event`` is recorded on the COMPUTE stream by fetch_free, after
    the last reader of the previous occupant; the transfer stream waits on it
    device-side before overwriting the buffer. That ordering is what lets the
    host submit ahead without ever blocking: the GPU enforces the recycle.
    """

    __slots__ = ("buffer", "free_event")

    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None
        self.free_event: torch.cuda.Event | None = None


_STATE_LOCK = threading.Lock()
_TICKETS: dict[int, _Ticket] = {}
_LIVE: collections.deque[int] = collections.deque()
_NEXT_ID = 0
_DEPTH = 3
_TRANSFER_STREAMS: dict[torch.device, torch.cuda.Stream] = {}
# Per-device ring of reusable device buffers, plus the indices currently
# available. A slot returns to _FREE_SLOTS when fetch_free SUBMITS (not when the
# GPU reaches it) -- the device-side wait on its free_event is what keeps the
# reuse correct, so the host never has to wait for compute to catch up.
_SLOTS: dict[torch.device, list[_Slot]] = {}
_FREE_SLOTS: dict[torch.device, collections.deque[int]] = {}
_STATS = {
    "fetches": 0,
    "bytes": 0,
    "copies": 0,
    "h2d_ms": 0.0,
    "wait_ms": 0.0,
    "depth_waits": 0,
}
_LIFETIME_STATS = dict(_STATS)
# (h2d_start, h2d_end) pairs awaiting timing. Drained only when the events have
# already completed, so accounting for a copy never blocks the host on it.
_PENDING_H2D: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

# Harness-only: restore the old behaviour of settling each copy's timing inside
# fetch_wait. Kept solely so a benchmark can A/B the cost of that host sync on
# the same build; production always drains lazily.
_BLOCKING_H2D_TIMING = False
_RUNTIME_OWNER_TOKEN = None


def set_h2d_timing_blocking(enabled: bool) -> None:
    global _BLOCKING_H2D_TIMING
    _BLOCKING_H2D_TIMING = bool(enabled)


def _drain_h2d(block: bool = False) -> None:
    """Accumulate h2d_ms for finished copies. Host-blocking only if block=True.

    block=True is for the end-of-run report, where the in-flight copies are done
    anyway; the hot path always calls this with block=False.
    """
    if not _PENDING_H2D:
        return
    pending = []
    for h2d_start, h2d_end in _PENDING_H2D:
        try:
            if not block and not h2d_end.query():
                pending.append((h2d_start, h2d_end))
                continue
            if block:
                h2d_end.synchronize()
            elapsed_ms = h2d_start.elapsed_time(h2d_end)
            _STATS["h2d_ms"] += elapsed_ms
            _LIFETIME_STATS["h2d_ms"] += elapsed_ms
        except RuntimeError:
            # Event never recorded (abandoned fetch, e.g. OOM unwind): drop it.
            pass
    _PENDING_H2D[:] = pending


def raise_dynamo_recompile_limit(min_limit: int = 128) -> None:
    """Lift dynamo's per-code-object recompile cap for the in-graph trunks.

    Two legitimate recompile sources stack up on one code object: bucketed
    training resolutions (dynamic=False -> one cache entry per distinct token
    shape) and sampling-boundary rebuilds (fresh block-fn closures fail the
    old entries' guards without evicting them). The default limit of 8 turned
    that into FailOnRecompileLimitHit at the third boundary of a 200-step run
    (~step 101). Each extra entry costs one ~3 min compile, not correctness;
    the cap exists to flag accidental recompile storms, which the boundary
    rebuild is not.
    """
    config = torch._dynamo.config
    for attribute in ("recompile_limit", "cache_size_limit"):
        current = getattr(config, attribute, None)
        if isinstance(current, int) and current < min_limit:
            setattr(config, attribute, min_limit)


def configure_fetch_runtime(*, depth: int = 3, owner_token=None) -> None:
    global _DEPTH, _NEXT_ID, _RUNTIME_OWNER_TOKEN
    if owner_token is not None:
        validate_process_owner(owner_token)
    if torch.cuda.is_available() and _SLOTS:
        # Slot buffers may still be in flight; settle before dropping them.
        torch.cuda.synchronize()
    _DEPTH = max(1, int(depth))
    with _STATE_LOCK:
        _TICKETS.clear()
        _LIVE.clear()
        _PENDING_H2D.clear()
        _SLOTS.clear()
        _FREE_SLOTS.clear()
        _NEXT_ID = 0
        _RUNTIME_OWNER_TOKEN = owner_token


def reset_fetch_stats() -> None:
    _PENDING_H2D.clear()
    for key in _STATS:
        _STATS[key] = 0


def fetch_stats(reset: bool = False) -> dict:
    # Settle the copies still in flight so the reported h2d_ms covers every
    # fetch, not just the ones that happened to finish before the last wait.
    _drain_h2d(block=True)
    stats = dict(_STATS)
    if reset:
        reset_fetch_stats()
    return stats


def lifetime_fetch_stats() -> dict:
    """Return monotonic fetch counters unaffected by report-window resets."""
    _drain_h2d(block=True)
    return dict(_LIFETIME_STATS)


def fetch_performance_metrics(stats: dict, *, step_wall_ms=None) -> dict:
    """Derive transfer-stream utilization from a settled reporting window.

    ``h2d_ms`` is CUDA-event time on the single serialized transfer stream.
    Dividing its window total by the matching step-wall total estimates transfer
    duty. ``wait_ms`` is deliberately excluded: it is host blocking around an
    event wait and does not say whether the GPU compute stream was idle.

    H2D timing drains opportunistically, so callers should provide a multi-step
    reporting window. Duty above 100% is retained and flagged rather than
    clamped; it indicates accounting carried across a window boundary or a
    mismatched denominator.
    """
    h2d_ms = float((stats or {}).get("h2d_ms", 0.0) or 0.0)
    byte_count = int((stats or {}).get("bytes", 0) or 0)
    wall_ms = None if step_wall_ms is None else float(step_wall_ms)
    duty_pct = None
    if wall_ms is not None and wall_ms > 0.0:
        duty_pct = 100.0 * h2d_ms / wall_ms
    achieved_gbps = None
    if h2d_ms > 0.0:
        achieved_gbps = byte_count / (h2d_ms * 1_000_000.0)
    return {
        "step_wall_ms": wall_ms,
        "h2d_duty_pct": duty_pct,
        "h2d_duty_overflow": bool(duty_pct is not None and duty_pct > 100.0),
        "achieved_gbps": achieved_gbps,
    }


def fetch_report(reset: bool = False, *, step_wall_ms=None) -> str | None:
    stats = fetch_stats(reset=reset)
    if not stats["fetches"]:
        return None
    metrics = fetch_performance_metrics(stats, step_wall_ms=step_wall_ms)
    gib = stats["bytes"] / 1024 ** 3
    duty = (
        "-" if metrics["h2d_duty_pct"] is None
        else f"{metrics['h2d_duty_pct']:.1f}"
    )
    gbps = (
        "-" if metrics["achieved_gbps"] is None
        else f"{metrics['achieved_gbps']:.2f}"
    )
    wall = (
        "-" if metrics["step_wall_ms"] is None
        else f"{metrics['step_wall_ms']:.3f}"
    )
    return (
        f"[InGraphStream] fetches={int(stats['fetches'])} "
        f"copies={int(stats['copies'])} "
        f"bytes={gib:.2f} GiB h2d_ms={stats['h2d_ms']:.3f} "
        f"step_wall_ms={wall} h2d_duty_pct={duty} "
        f"h2d_duty_overflow={int(metrics['h2d_duty_overflow'])} "
        f"achieved_gbps={gbps} "
        f"wait_ms={stats['wait_ms']:.3f} depth_waits={int(stats['depth_waits'])}"
    )


def _transfer_stream(device: torch.device):
    stream = _TRANSFER_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _TRANSFER_STREAMS[device] = stream
    return stream


def _ring_locked(device: torch.device) -> tuple[list[_Slot], collections.deque]:
    slots = _SLOTS.get(device)
    if slots is None or len(slots) != _DEPTH:
        slots = [_Slot() for _ in range(_DEPTH)]
        _SLOTS[device] = slots
        _FREE_SLOTS[device] = collections.deque(range(_DEPTH))
    return slots, _FREE_SLOTS[device]


def _acquire_slot(device: torch.device, nbytes: int) -> tuple[int, _Slot]:
    """Take a ring slot and make sure its buffer holds nbytes.

    Never blocks on GPU progress. An empty free list means the graph is holding
    more than `depth` fetched buffers live at once -- the same condition the old
    host-side reaper raised on, and still a bug rather than something to wait
    out (waiting here would mean waiting on the compute stream, which only the
    host can feed).
    """
    with _STATE_LOCK:
        slots, free = _ring_locked(device)
        if not free:
            raise RuntimeError(
                "mm.fetch_start depth exceeded before fetch_free "
                f"(ring depth {_DEPTH}); the graph holds more fetched buffers "
                "live than the ring has slots"
            )
        index = free.popleft()
    slot = slots[index]
    if slot.buffer is None or slot.buffer.numel() < nbytes:
        # Growth happens during warmup, until every slot has seen the largest
        # block. Settle the device first: the outgoing buffer may still be in
        # flight, and dropping its last reference would hand the memory back to
        # the caching allocator while a stream is still reading it.
        if slot.buffer is not None:
            torch.cuda.synchronize(device)
        slot.buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
    return index, slot


def _release_slot(device: torch.device, index: int) -> None:
    with _STATE_LOCK:
        _FREE_SLOTS.setdefault(device, collections.deque()).append(index)


def drain_fetch_runtime(*, owner_token=None) -> int:
    """Abandon every outstanding fetch ticket (OOM-recovery path only).

    An OOM unwinds a forward between fetch_start and fetch_free, leaving
    tickets whose free_event never records; the next fetch_start then blocks
    on the depth limit and raises 'depth exceeded before fetch_free'. The
    recovery path (mid-denoise demote / full streamed transition) calls this
    AFTER the failed forward has fully unwound: nothing will consume the
    in-flight device buffers anymore, so waiting out the transfer streams and
    dropping the tickets is safe. Returns the number of tickets abandoned.
    """
    token = _RUNTIME_OWNER_TOKEN if owner_token is None else owner_token
    if token is not None:
        validate_process_owner(token)
    with _STATE_LOCK:
        for stream in _TRANSFER_STREAMS.values():
            stream.synchronize()
        abandoned = len(_LIVE)
        _LIVE.clear()
        _TICKETS.clear()
        _PENDING_H2D.clear()
        # The abandoned tickets never called fetch_free, so their slots were
        # never returned. Rebuild the free list and clear the stale free_events
        # (the recovery path has already unwound whatever would have read them).
        for device, slots in _SLOTS.items():
            for slot in slots:
                slot.free_event = None
            _FREE_SLOTS[device] = collections.deque(range(len(slots)))
    return abandoned


def release_fetch_runtime(owner_token) -> None:
    """Release ring, streams, events, tickets, and reporting state for owner."""
    global _NEXT_ID, _RUNTIME_OWNER_TOKEN
    validate_process_owner(owner_token)
    drain_fetch_runtime(owner_token=owner_token)
    with _STATE_LOCK:
        _drain_h2d(block=True)
        _TICKETS.clear()
        _LIVE.clear()
        _PENDING_H2D.clear()
        _SLOTS.clear()
        _FREE_SLOTS.clear()
        _TRANSFER_STREAMS.clear()
        _NEXT_ID = 0
        for key in _STATS:
            _STATS[key] = 0
            _LIFETIME_STATS[key] = 0
        _RUNTIME_OWNER_TOKEN = None


def _fetch_start_impl(host_flat: torch.Tensor) -> torch.Tensor:
    if _RUNTIME_OWNER_TOKEN is not None:
        validate_process_owner(_RUNTIME_OWNER_TOKEN)
    if host_flat.device.type != "cpu":
        raise RuntimeError("mm.fetch_start expected a CPU host_flat tensor")
    if torch.cuda.is_available() and not pin_manager.is_host_pinned(host_flat):
        # is_host_pinned, not host_flat.is_pinned(): arena flats are pinned
        # in place with cudaHostRegister, which torch's is_pinned() does not
        # recognize (it only tracks its own caching-allocator pins).
        raise RuntimeError("mm.fetch_start expected a pinned host_flat tensor")
    device = torch.device("cuda")
    stream = _transfer_stream(device)
    nbytes = host_flat.numel()
    index, slot = _acquire_slot(device, nbytes)
    with _STATE_LOCK:
        global _NEXT_ID
        tid = _NEXT_ID
        _NEXT_ID += 1
        _LIVE.append(tid)
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    ready = torch.cuda.Event()
    device_buffer = slot.buffer[:nbytes]
    with torch.cuda.stream(stream):
        if slot.free_event is not None:
            # Device-side recycle: the copy waits for the previous occupant's
            # last reader, so the host does not have to.
            stream.wait_event(slot.free_event)
        h2d_start.record(stream)
        device_buffer.copy_(host_flat, non_blocking=True)
        ready.record(stream)
        h2d_end.record(stream)
    with _STATE_LOCK:
        _TICKETS[tid] = _Ticket(
            tid=tid,
            device_buffer=device_buffer,
            ready_event=ready,
            slot=index,
            slot_device=device,
            h2d_start=h2d_start,
            h2d_end=h2d_end,
            nbytes=nbytes,
            copies=1,
        )
        _STATS["fetches"] += 1
        _STATS["bytes"] += int(host_flat.numel())
        _STATS["copies"] += 1
        _LIFETIME_STATS["fetches"] += 1
        _LIFETIME_STATS["bytes"] += int(host_flat.numel())
        _LIFETIME_STATS["copies"] += 1
    return torch.tensor([tid], dtype=torch.int64)


def _validated_transfer_ranges(
    host_flat: torch.Tensor,
    ranges: torch.Tensor,
    compact_nbytes: int,
) -> list[tuple[int, int, int]]:
    """Validate a static multi-range plan at the opaque runtime boundary.

    Ranges are tensor data rather than a closed-over Python plan so replacing
    same-shaped canonical storage remains guard-stable. Destination spans must
    form one dense compact flat; source spans must be ordered, non-overlapping
    canonical bytes.
    """
    if host_flat.device.type != "cpu" or host_flat.dtype != torch.uint8:
        raise RuntimeError("mm.fetch_start_multi expected a CPU uint8 host_flat")
    if not host_flat.is_contiguous():
        raise RuntimeError("mm.fetch_start_multi expected a contiguous host_flat")
    if not pin_manager.is_arena_backed(host_flat):
        raise RuntimeError(
            "mm.fetch_start_multi expected a registered canonical arena source"
        )
    if ranges.device.type != "cpu" or ranges.dtype != torch.int64:
        raise RuntimeError("mm.fetch_start_multi expected CPU int64 ranges")
    if ranges.ndim != 2 or ranges.shape[1] != 3 or ranges.shape[0] == 0:
        raise RuntimeError("mm.fetch_start_multi expected non-empty Nx3 ranges")
    compact_nbytes = int(compact_nbytes)
    if compact_nbytes <= 0:
        raise RuntimeError("mm.fetch_start_multi expected compact_nbytes > 0")

    rows = [tuple(int(value) for value in row) for row in ranges.tolist()]
    previous_src_end = 0
    expected_dst = 0
    for index, (src_offset, dst_offset, nbytes) in enumerate(rows):
        if src_offset < 0 or dst_offset < 0 or nbytes <= 0:
            raise RuntimeError(f"mm.fetch_start_multi invalid range {index}")
        if src_offset + nbytes > host_flat.numel():
            raise RuntimeError(f"mm.fetch_start_multi source range {index} out of bounds")
        if index and src_offset < previous_src_end:
            raise RuntimeError(f"mm.fetch_start_multi source range {index} overlaps")
        if dst_offset != expected_dst:
            raise RuntimeError(
                f"mm.fetch_start_multi destination range {index} is not compact"
            )
        previous_src_end = src_offset + nbytes
        expected_dst = dst_offset + nbytes
    if expected_dst != compact_nbytes:
        raise RuntimeError("mm.fetch_start_multi ranges do not fill compact_nbytes")
    return rows


def _fetch_start_multi_impl(
    host_flat: torch.Tensor,
    ranges: torch.Tensor,
    compact_nbytes: int,
) -> torch.Tensor:
    rows = _validated_transfer_ranges(host_flat, ranges, compact_nbytes)
    device = torch.device("cuda")
    stream = _transfer_stream(device)
    compact_nbytes = int(compact_nbytes)
    index, slot = _acquire_slot(device, compact_nbytes)
    with _STATE_LOCK:
        global _NEXT_ID
        tid = _NEXT_ID
        _NEXT_ID += 1
        _LIVE.append(tid)

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    ready = torch.cuda.Event()
    device_buffer = slot.buffer[:compact_nbytes]
    with torch.cuda.stream(stream):
        if slot.free_event is not None:
            stream.wait_event(slot.free_event)
        h2d_start.record(stream)
        for src_offset, dst_offset, nbytes in rows:
            device_buffer[dst_offset:dst_offset + nbytes].copy_(
                host_flat[src_offset:src_offset + nbytes], non_blocking=True
            )
        ready.record(stream)
        h2d_end.record(stream)

    with _STATE_LOCK:
        _TICKETS[tid] = _Ticket(
            tid=tid,
            device_buffer=device_buffer,
            ready_event=ready,
            slot=index,
            slot_device=device,
            h2d_start=h2d_start,
            h2d_end=h2d_end,
            nbytes=compact_nbytes,
            copies=len(rows),
        )
        _STATS["fetches"] += 1
        _STATS["bytes"] += compact_nbytes
        _STATS["copies"] += len(rows)
        _LIFETIME_STATS["fetches"] += 1
        _LIFETIME_STATS["bytes"] += compact_nbytes
        _LIFETIME_STATS["copies"] += len(rows)
    return torch.tensor([tid], dtype=torch.int64)


@torch.library.custom_op("mm::fetch_start", mutates_args=())
def fetch_start(host_flat: torch.Tensor) -> torch.Tensor:
    return _fetch_start_impl(host_flat)


@fetch_start.register_fake
def _(host_flat):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_start_after", mutates_args=("guard",))
def fetch_start_after(host_flat: torch.Tensor, guard: torch.Tensor) -> torch.Tensor:
    return _fetch_start_impl(host_flat)


@fetch_start_after.register_fake
def _(host_flat, guard):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_start_gated", mutates_args=())
def fetch_start_gated(host_flat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """fetch_start with a REAL data dependency on `gate`.

    fetch_start_after's guard is a declared mutation; that bookkeeping does
    not survive Inductor's scheduler/reinplacer, which may hoist backward
    re-fetches above frees (ring overrun). Here the gate is an ordinary
    input, so the dependency is genuine dataflow no scheduling stage can
    drop. Emitted by the post-grad ordering pass (ingraph_stream_scheduling);
    not intended for hand-written model code."""
    return _fetch_start_impl(host_flat)


@fetch_start_gated.register_fake
def _(host_flat, gate):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_start_multi", mutates_args=())
def fetch_start_multi(
    host_flat: torch.Tensor, ranges: torch.Tensor, compact_nbytes: int
) -> torch.Tensor:
    return _fetch_start_multi_impl(host_flat, ranges, compact_nbytes)


@fetch_start_multi.register_fake
def _(host_flat, ranges, compact_nbytes: int):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_start_multi_after", mutates_args=("guard",))
def fetch_start_multi_after(
    host_flat: torch.Tensor,
    ranges: torch.Tensor,
    compact_nbytes: int,
    guard: torch.Tensor,
) -> torch.Tensor:
    return _fetch_start_multi_impl(host_flat, ranges, compact_nbytes)


@fetch_start_multi_after.register_fake
def _(host_flat, ranges, compact_nbytes: int, guard):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_start_multi_gated", mutates_args=())
def fetch_start_multi_gated(
    host_flat: torch.Tensor,
    ranges: torch.Tensor,
    compact_nbytes: int,
    gate: torch.Tensor,
) -> torch.Tensor:
    return _fetch_start_multi_impl(host_flat, ranges, compact_nbytes)


@fetch_start_multi_gated.register_fake
def _(host_flat, ranges, compact_nbytes: int, gate):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_wait", mutates_args=())
def fetch_wait(token: torch.Tensor, nbytes: int) -> torch.Tensor:
    tid = int(token[0].item())
    ticket = _TICKETS.get(tid)
    if ticket is None:
        raise RuntimeError(f"mm.fetch_wait got unknown ticket {tid}")
    if int(nbytes) != ticket.nbytes:
        raise RuntimeError(
            f"mm.fetch_wait size mismatch for ticket {tid}: "
            f"expected {ticket.nbytes}, got {int(nbytes)}"
        )
    current = torch.cuda.current_stream()
    start = time.perf_counter()
    current.wait_event(ticket.ready_event)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _STATS["wait_ms"] += elapsed_ms
    _LIFETIME_STATS["wait_ms"] += elapsed_ms
    if ticket.h2d_start is not None and ticket.h2d_end is not None:
        # Queue the pair for opportunistic draining; do NOT synchronize here.
        # Blocking on h2d_end just to service a counter stalls the submit loop,
        # and with the ring recycling device-side there is nothing to gain from
        # the throttle it used to provide.
        _PENDING_H2D.append((ticket.h2d_start, ticket.h2d_end))
        _drain_h2d(block=_BLOCKING_H2D_TIMING)
    return ticket.device_buffer


@fetch_wait.register_fake
def _(token, nbytes: int):
    return torch.empty(nbytes, dtype=torch.uint8, device="cuda")


def _fetch_free_impl(token: torch.Tensor) -> torch.Tensor:
    tid = int(token[0].item())
    ticket = _TICKETS.get(tid)
    if ticket is None:
        raise RuntimeError(f"mm.fetch_free got unknown ticket {tid}")
    free_event = torch.cuda.Event()
    free_event.record(torch.cuda.current_stream())
    ticket.free_event = free_event
    if ticket.slot >= 0 and ticket.slot_device is not None:
        # The slot's next occupant waits on this event device-side before it
        # overwrites the buffer, so the slot can go back into circulation now,
        # at SUBMIT time, without the host waiting for the GPU to reach it.
        _SLOTS[ticket.slot_device][ticket.slot].free_event = free_event
        _release_slot(ticket.slot_device, ticket.slot)
    with _STATE_LOCK:
        _TICKETS.pop(tid, None)
        if _LIVE and _LIVE[0] == tid:
            _LIVE.popleft()
        elif tid in _LIVE:
            _LIVE.remove(tid)
    return token.clone()


@torch.library.custom_op("mm::fetch_free", mutates_args=())
def fetch_free(token: torch.Tensor) -> torch.Tensor:
    return _fetch_free_impl(token)


@fetch_free.register_fake
def _(token):
    return token.clone()


@torch.library.custom_op("mm::fetch_free_after", mutates_args=("guard",))
def fetch_free_after(token: torch.Tensor, guard: torch.Tensor) -> torch.Tensor:
    return _fetch_free_impl(token)


@fetch_free_after.register_fake
def _(token, guard):
    return token.clone()


def _register_ordered_effects():
    """Pin the fetch ops to program order inside compiled graphs.

    Functionalized custom ops only carry data deps through their args; in the
    AOT backward graph each checkpoint unit's re-fetch depends only on the
    saved boundary activation (an immediately-available input), so Inductor
    may hoist all re-fetches above the frees -- exceeding the ring depth and
    deadlocking the host-side depth guard. Ordered effect tokens thread a
    dependency chain through every fetch op, enforcing eager program order in
    forward AND backward graphs (kernel-launch order only; stream overlap is
    unaffected)."""
    try:
        from torch._higher_order_ops.effects import (
            _EffectType,
            _register_effectful_op,
        )

        for op in (
            torch.ops.mm.fetch_start.default,
            torch.ops.mm.fetch_start_after.default,
            torch.ops.mm.fetch_start_multi.default,
            torch.ops.mm.fetch_start_multi_after.default,
            torch.ops.mm.fetch_wait.default,
            torch.ops.mm.fetch_free.default,
            torch.ops.mm.fetch_free_after.default,
        ):
            _register_effectful_op(op, _EffectType.ORDERED)
    except Exception as error:  # pragma: no cover - torch-version dependent
        raise RuntimeError(
            "in-graph streaming requires ordered-effect registration for its "
            f"fetch ops (torch internal API changed?): {error!r}"
        ) from error


# NOT registered at import time: in torch 2.12 ordered-effect tokens trip an
# internal token-erasure assertion inside the checkpoint HOP lowering
# (see tests/test_ingraph_training_ops.py, compiled xfail). Phase 4a S1 keeps
# this as the candidate ordering mechanism for the compiled trunk; call it
# explicitly once the HOP interaction is resolved (torch upgrade or flat-trunk
# design without the checkpoint HOP).


class _FreeOnBackwardFn(torch.autograd.Function):
    """Anchor a ticket's free event to the consuming block's BACKWARD.

    Training-mode counterpart of `fetch_free_after`: under checkpoint
    recompute, the block's backward (grad-input from the fetched weight
    views) is the true last reader of the ticket's device buffer, so a
    forward-side free lets the depth-K ring recycle the buffer under
    backward kernels that are still reading it (silent corruption).

    Wrap the block INPUT, not its output: this node's backward runs last
    in the block's backward (input side), i.e. after every weight-view
    read, and `fetch_free_after`'s declared guard mutation on the incoming
    grad keeps the free ordered after the kernels that produced it.
    """

    @staticmethod
    def forward(ctx, x, token):
        ctx.save_for_backward(token)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        (token,) = ctx.saved_tensors
        if torch.compiler.is_compiling():
            # Declared guard mutation orders the free after the kernels that
            # produced grad_x; functionalization makes it version-safe.
            torch.ops.mm.fetch_free_after(token, grad_x)
        else:
            # Eager executes in program order -- and the guarded variant's
            # version bump on grad_x would trip autograd's version checks.
            torch.ops.mm.fetch_free(token)
        return grad_x, None


def free_on_backward(x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    """Defer a ticket's free to the consuming block's backward (training).

    Two fetch generations exist under non-reentrant checkpoint: the
    first-pass fetch (its views are dropped by the checkpoint hooks, so it
    is safe to free after the block's forward) and the recompute fetch
    (its views feed the real backward, so it must be freed after the
    block's backward). The token passed here is saved via
    ``save_for_backward`` -- checkpoint's saved-tensor machinery therefore
    swaps it for the RECOMPUTE generation's token automatically, and this
    node's backward frees exactly the ticket backward actually read.

    Canonical training block shape (see checkpoint_recompute_context):

        token = fetch_start_after(host, x)
        flat = fetch_wait(token, nbytes)
        ...views...
        x = free_on_backward(x, token)
        out = <block math>(x, views)
        if not in_recompute():
            torch.ops.mm.fetch_free_after(token, out)  # first-pass gen only
        return out
    """
    return _FreeOnBackwardFn.apply(x, token)


_IN_RECOMPUTE = threading.local()


def in_recompute() -> bool:
    """True while a checkpoint recompute pass (via checkpoint_recompute_context)
    is re-running the block fn."""
    return bool(getattr(_IN_RECOMPUTE, "value", False))


class _RecomputeMarker:
    def __enter__(self):
        self._prev = getattr(_IN_RECOMPUTE, "value", False)
        _IN_RECOMPUTE.value = True
        return self

    def __exit__(self, exc_type, exc, tb):
        _IN_RECOMPUTE.value = self._prev
        return False


def checkpoint_recompute_context():
    """``context_fn`` for torch.utils.checkpoint: null forward context, and a
    recompute context that flips in_recompute() so the block fn suppresses the
    first-pass forward free during recompute (the recompute ticket is freed by
    free_on_backward instead). EAGER ONLY -- compiled checkpoint requires
    TorchDispatchMode contexts; use compiled_checkpoint_context there."""
    return contextlib.nullcontext(), _RecomputeMarker()


def _compiled_free_policy(ctx, op, *args, **kwargs):
    from torch.utils.checkpoint import CheckpointPolicy

    if op in (
        torch.ops.mm.fetch_free_after.default,
        torch.ops.mm.fetch_free.default,
    ):
        # Keep the forward-side free OUT of the backward replay: replayed, it
        # would free the backward re-fetch's buffer before the grad kernels
        # read it. free_on_backward's op is the backward-side free.
        return CheckpointPolicy.MUST_SAVE
    if op in (
        torch.ops.mm.fetch_start.default,
        torch.ops.mm.fetch_start_after.default,
        torch.ops.mm.fetch_start_multi.default,
        torch.ops.mm.fetch_start_multi_after.default,
        torch.ops.mm.fetch_wait.default,
    ):
        # The design's core invariant: fetched weights are NEVER saved for
        # backward. PREFER_RECOMPUTE is advisory -- at Krea2 scale the
        # partitioner chose to save all 28 fetched flats (12.25 GiB -> OOM).
        return CheckpointPolicy.MUST_RECOMPUTE
    # Everything else replays in backward (full-checkpoint mode).
    return CheckpointPolicy.PREFER_RECOMPUTE


def compiled_checkpoint_context():
    """``context_fn`` for torch.utils.checkpoint under torch.compile."""
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return create_selective_checkpoint_contexts(_compiled_free_policy)


# NOTE: there is deliberately NO helper that "picks the right checkpoint
# context automatically". The checkpoint HOP calls context_fn() OUTSIDE the
# compiling frame, so an is_compiling() check inside such a helper always
# reads False under compile and hands the HOP eager (non-TorchDispatchMode)
# contexts, failing its assertion. Select the context at trunk level instead:
#     context_fn = (compiled_checkpoint_context
#                   if torch.compiler.is_compiling()
#                   else checkpoint_recompute_context)
