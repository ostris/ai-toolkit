"""Dedicated-VRAM (WDDM) budget arithmetic: one source of truth, one name per meaning.

This module owns the *dedicated-cliff* quantities. Windows/WDDM has two distinct
memory cliffs with different failure modes:

* **Dedicated ceiling** (this module): crossing the card's physical VRAM makes
  WDDM silently page GPU memory to system RAM -- catastrophic slowdown, no
  error. Governed by ``device_free_bytes`` below -- NOT by
  ``torch.cuda.mem_get_info``, which does *not* see other processes (see the
  note on the free signal). Quantities: ``hard_gib`` (the never-cross
  device-free floor, default 1.0), ``margin_gib`` (the planning headroom,
  >= hard).
* **Shared / DXGI NON_LOCAL budget** (NOT this module): pinned host memory
  commits against it and exhausting it is a hard cudaErrorMemoryAllocation.
  That reserve lives in ``pin_manager`` / ``bounce_pool``
  (``dxgi_spill_reserve_bytes``); do not conflate the two.

The free signal (do not regress this)
-------------------------------------
``torch.cuda.mem_get_info`` free is NOT physical availability. It is what the
driver will promise *this process*; on WDDM that promise is backed by paging
other processes out. Measured on the 4070 with a second process holding 9 GiB:

    NVML physical free              : 0.55 GiB   <- the truth
    torch.cuda.mem_get_info free    : 4.70 GiB   <- over-reports ~8x
    DXGI LOCAL Budget - CurrentUsage: 4.58 GiB   <- also over-reports

Planning against the optimistic number is precisely how a run silently crosses
the dedicated cliff (a 73 s/it job ran at 304 s/it with an orphan process on the
card, and no allocator retry fired). So ``free`` here means NVML physical free
whenever NVML is available -- see ``device_free_bytes``. DXGI stays where it
belongs: the *shared* NON_LOCAL pin budget, not the dedicated cliff.

Vocabulary (each quantity has exactly one name):

* ``total`` / ``free`` -- physical card bytes, via ``device_free_bytes``
  (NVML-backed; sees every process on the GPU).
* ``torch_reserved`` / ``torch_allocated`` -- torch's caching allocator.
* ``non_torch`` -- ``(total - free) - torch_reserved``: CUDA context, VAE/TE,
  the Windows desktop, other processes. Torch is never the card's only tenant.
* ``hard`` -- device-free floor the card must keep (WDDM spill guard).
* ``margin`` -- planning headroom subtracted from budgets; ``>= hard``.

Everything here is pure (CPU-testable) except ``DeviceSnapshot.capture`` and
``device_free_bytes``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from . import nvml_meminfo

GIB = 1024 ** 3


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def reconcile_free_bytes(driver_free_bytes, physical_free_bytes) -> int:
    """Pick the governing device-free value (pure; CPU-testable).

    ``physical_free_bytes`` is NVML's device-wide free (None when NVML is
    unavailable). We take the MIN of the two signals rather than trusting NVML
    blindly: the driver number is a real constraint too, and whichever sensor is
    more pessimistic is the one that keeps us off the cliff. In the uncontended
    single-process case the two agree, so this does not cost residency.
    """
    driver = max(0, int(driver_free_bytes))
    if physical_free_bytes is None:
        return driver
    return min(driver, max(0, int(physical_free_bytes)))


# --------------------------------------------------------------------------
# Simulated smaller card (validation knob)
# --------------------------------------------------------------------------
# Pretend the GPU is smaller than it is, so an 8 GB / 6 GB card's residency,
# streaming and cap behaviour can be exercised on a 12 GB one. Implemented as a
# *phantom ballast*: a fixed number of bytes subtracted from BOTH total and free
# everywhere this module reports them. That keeps every derived quantity
# self-consistent (non_torch, margins, promotion checks, the allocator cap) with
# no per-call-site special cases -- an unfittable model is unfittable for the
# same arithmetic reason it would be on the real small card.
#
# It does not shrink the physical card: the allocator cap is what actually makes
# an over-plan fail, and `_apply_wddm_hard_allocator_cap` converts the simulated
# cap bytes back into a fraction of the REAL total before handing it to torch.
_SIMULATED_CARD_BYTES: int | None = None


def set_simulated_card_bytes(total_bytes) -> None:
    """Pretend the card has ``total_bytes`` of VRAM (``None`` disables)."""
    global _SIMULATED_CARD_BYTES
    if total_bytes is None:
        _SIMULATED_CARD_BYTES = None
        return
    value = int(total_bytes)
    if value <= 0:
        raise ValueError(f"simulated card size must be positive, got {total_bytes}")
    _SIMULATED_CARD_BYTES = value


def simulated_card_bytes() -> int | None:
    return _SIMULATED_CARD_BYTES


def apply_simulated_card(simulated_vram_gib, *, device=None) -> int | None:
    """Install (or clear) the simulated card size; logs once when active.

    ``simulated_vram_gib`` of ``None``/``0`` clears the simulation. A size at or
    above the real card is a no-op with a warning -- the ballast can only hide
    VRAM, never invent it.
    """
    if not simulated_vram_gib:
        set_simulated_card_bytes(None)
        return None

    wanted = int(float(simulated_vram_gib) * GIB)
    if device is None or not torch.cuda.is_available():
        set_simulated_card_bytes(wanted)
        return wanted

    real = real_device_total_bytes(device)
    if wanted >= real:
        print(
            "[MemoryManager] simulated VRAM "
            f"{wanted / GIB:.2f} GiB >= real card {real / GIB:.2f} GiB; "
            "ignoring (a simulation can only shrink the card)"
        )
        set_simulated_card_bytes(None)
        return None

    set_simulated_card_bytes(wanted)
    print(
        "[MemoryManager] SIMULATED CARD: reporting "
        f"{wanted / GIB:.2f} GiB total (real {real / GIB:.2f} GiB); "
        f"{(real - wanted) / GIB:.2f} GiB is hidden from both total and free, "
        "and the allocator cap enforces it. Planning, residency and OOMs now "
        "behave as they would on the smaller card."
    )
    return wanted


def real_device_total_bytes(device) -> int:
    """Physical card size, ignoring any simulation."""
    return int(torch.cuda.mem_get_info(device)[1])


def simulated_ballast_bytes(device) -> int:
    """Bytes hidden from total AND free to fake a smaller card (0 when off)."""
    if _SIMULATED_CARD_BYTES is None:
        return 0
    return max(0, real_device_total_bytes(device) - _SIMULATED_CARD_BYTES)


def device_total_bytes(device) -> int:
    """Governing card size: the simulated one when a simulation is active."""
    return max(0, real_device_total_bytes(device) - simulated_ballast_bytes(device))


def device_free_bytes(device) -> int:
    """Governing physical free bytes on ``device``, across ALL processes.

    Prefer this over ``torch.cuda.mem_get_info(...)[0]`` anywhere a decision is
    made (residency, promotion, reserve, cap). ``mem_get_info`` alone is blind to
    other GPU tenants and will happily plan a run onto memory that is not there.
    Delta/probe loops that measure this process's *own* allocation deltas may
    keep using the raw driver number -- they are measuring differences, not
    availability.
    """
    driver_free = int(torch.cuda.mem_get_info(device)[0])
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    free = reconcile_free_bytes(
        driver_free, nvml_meminfo.physical_free_bytes(index)
    )
    return max(0, free - simulated_ballast_bytes(device))


# A quiet box still has non-torch bytes on the card: the CUDA context, cuDNN /
# cuBLAS workspaces, and the Windows desktop compositor. Measured ~1.15 GiB on
# the 4070 with a bare context, so treat everything under this as the cost of
# doing business rather than "contention".
FOREIGN_NOMINAL_BYTES = int(1.5 * GIB)
# Do not cry about a browser tab. Only excess above this is worth a line in the
# log; below it, the residency we lose is not what makes or breaks a run.
FOREIGN_WARN_FLOOR_BYTES = int(1.0 * GIB)
# Above this, a foreign tenant is doing real damage even if the model could
# never have been fully resident anyway: every GiB it holds is a GiB we stream
# from the CPU on every single step. Big models (Krea2 on a 12 GB card) never
# fit fully, so "would it have fit?" must NOT be the thing that decides whether
# we raise our voice -- lost residency is.
FOREIGN_SEVERE_BYTES = int(2.0 * GIB)


@dataclass(frozen=True)
class ForeignVramReport:
    """Is someone ELSE on this GPU, and is that why we cannot fit?

    ``foreign`` is every byte on the card that is not our torch allocator --
    other processes, plus our own context/desktop baseline. ``excess`` is what
    remains after allowing a nominal baseline, i.e. the part plausibly caused by
    another tenant (a game, a ComfyUI server, an orphaned training job).
    """

    foreign_bytes: int
    excess_bytes: int
    want_bytes: int
    have_bytes: int
    fits_now: bool
    fits_without_excess: bool
    severity: str  # "none" | "benign" | "contributing" | "blocking"

    @property
    def is_blocking(self) -> bool:
        """The model would have fit; another tenant is the reason it does not."""
        return self.severity == "blocking"

    def format(self) -> str:
        return (
            f"foreign={self.foreign_bytes / GIB:.2f} GiB "
            f"(excess={self.excess_bytes / GIB:.2f} GiB) "
            f"want={self.want_bytes / GIB:.2f} GiB "
            f"have={self.have_bytes / GIB:.2f} GiB "
            f"severity={self.severity}"
        )


def assess_foreign_vram(
    *,
    total_bytes,
    free_bytes,
    torch_reserved_bytes,
    want_bytes,
    have_bytes,
    nominal_bytes: int = FOREIGN_NOMINAL_BYTES,
    warn_floor_bytes: int = FOREIGN_WARN_FLOOR_BYTES,
) -> ForeignVramReport:
    """Classify external VRAM pressure at a phase boundary (pure, CPU-testable).

    ``want_bytes``  -- bytes to hold the model fully resident.
    ``have_bytes``  -- residency budget we actually got.
    ``free_bytes``  -- must be the NVML-backed physical free (``device_free_bytes``);
                       with ``mem_get_info`` the foreign bytes are invisible, which
                       is the whole reason this check exists.

    The verdict is a counterfactual, not a threshold on usage alone: contention
    only earns a warning when it *changed the outcome*.

      blocking     -- we are streaming, but would have fit without the excess.
                      Someone else's memory is costing us real throughput.
      contributing -- the excess costs residency, but the model would not have
                      fit anyway. Worth a note; not the root cause.
      benign       -- another tenant is present, but we fit regardless.
      none         -- nothing meaningful beyond our own baseline.
    """
    used = max(0, int(total_bytes) - int(free_bytes))
    foreign = max(0, used - max(0, int(torch_reserved_bytes)))
    excess = max(0, foreign - max(0, int(nominal_bytes)))

    want = max(0, int(want_bytes))
    have = max(0, int(have_bytes))
    fits_now = have >= want
    fits_without_excess = (have + excess) >= want

    if excess < max(0, int(warn_floor_bytes)):
        severity = "none"
    elif fits_now:
        severity = "benign"
    elif fits_without_excess:
        severity = "blocking"
    else:
        severity = "contributing"

    return ForeignVramReport(
        foreign_bytes=foreign,
        excess_bytes=excess,
        want_bytes=want,
        have_bytes=have,
        fits_now=fits_now,
        fits_without_excess=fits_without_excess,
        severity=severity,
    )


def format_foreign_vram_warning(report: ForeignVramReport, *, phase: str) -> str | None:
    """Operator-facing line for a phase boundary. None when there is nothing to say.

    Volume is set by HARM, not by the fit counterfactual: a model too big to ever
    be fully resident (Krea2 on 12 GB) still loses a GiB of residency for every
    GiB a foreign tenant holds, and streams it from the CPU every step.
    """
    if report.severity in ("none", "benign"):
        return None

    gib = GIB
    excess = report.excess_bytes / gib
    culprits = (
        "Check for another training job, a leftover/orphaned run of this same job, "
        "a ComfyUI server, or a game."
    )
    head = (
        f"{excess:.2f} GiB of this card is held outside this process "
        f"({report.foreign_bytes / gib:.2f} GiB total non-torch), costing an equal "
        f"amount of resident weights during {phase} -- those are streamed from the "
        f"CPU on every step instead."
    )

    if report.severity == "blocking":
        return (
            f"[MemoryManager] WARNING: VRAM contention is why {phase} is streaming. "
            f"{head} The model needs {report.want_bytes / gib:.2f} GiB resident and "
            f"only {report.have_bytes / gib:.2f} GiB is available -- it WOULD fit "
            f"entirely if that memory were free. {culprits}"
        )
    if report.excess_bytes >= FOREIGN_SEVERE_BYTES:
        return (
            f"[MemoryManager] WARNING: heavy VRAM contention during {phase}. {head} "
            f"(The model, {report.want_bytes / gib:.2f} GiB, would not be fully "
            f"resident even on an idle card, but this is still costing real "
            f"throughput.) {culprits}"
        )
    return (
        f"[MemoryManager] note: {head} {culprits}"
    )


def device_mem_info(device) -> tuple[int, int]:
    """Drop-in for ``torch.cuda.mem_get_info``: ``(free, total)`` in bytes.

    Identical shape to the torch call, but ``free`` is the NVML-backed physical
    free (sees other processes) and ``total`` honours a simulated smaller card.
    """
    return device_free_bytes(device), device_total_bytes(device)


@dataclass(frozen=True)
class DeviceSnapshot:
    """Point-in-time physical view of a CUDA device (bytes)."""

    total: int
    free: int
    torch_reserved: int
    torch_allocated: int

    @property
    def used(self) -> int:
        return max(0, self.total - self.free)

    @property
    def non_torch(self) -> int:
        """Device bytes held by anyone but torch's caching allocator."""
        return max(0, self.used - self.torch_reserved)

    @staticmethod
    def capture(device) -> Optional["DeviceSnapshot"]:
        if device is None or not torch.cuda.is_available():
            return None
        dev = torch.device(device)
        if dev.type != "cuda":
            return None
        return DeviceSnapshot(
            total=device_total_bytes(dev),
            # NVML-backed: sees other processes on the card. See the module
            # docstring -- mem_get_info free would over-report here.
            free=device_free_bytes(dev),
            torch_reserved=int(torch.cuda.memory_reserved(dev)),
            torch_allocated=int(torch.cuda.memory_allocated(dev)),
        )

    def format(self) -> str:
        return (
            f"torch_allocated={self.torch_allocated / GIB:.2f} GiB "
            f"torch_reserved={self.torch_reserved / GIB:.2f} GiB "
            f"device_used={self.used / GIB:.2f}/{self.total / GIB:.2f} GiB "
            f"device_free={self.free / GIB:.2f} GiB "
            f"non_torch={self.non_torch / GIB:.2f} GiB"
        )


@dataclass(frozen=True)
class WddmMargins:
    """Dedicated-cliff margins for one phase (training attach / sampling start).

    Resolve ONCE at the phase boundary and pass by value; do not re-read env
    vars mid-phase (they cannot change mid-run, and re-reads hide which value
    actually governed a decision).
    """

    hard_gib: float
    margin_gib: float
    source: str  # "config" | "env" | "auto"

    @property
    def hard_bytes(self) -> int:
        return int(self.hard_gib * GIB)

    @property
    def margin_bytes(self) -> int:
        return int(self.margin_gib * GIB)

    def format(self) -> str:
        return (
            f"wddm_hard={self.hard_gib:.2f} GiB "
            f"wddm_margin={self.margin_gib:.2f} GiB ({self.source})"
        )


def auto_margin_gib(device, pct: float = 0.10, floor_gib: float = 1.0) -> float:
    """Auto planning margin: max(floor, pct * card size)."""
    try:
        total_bytes = device_total_bytes(device)
    except Exception:
        total_bytes = 0
    total_gib = max(0.0, float(total_bytes) / GIB)
    return max(float(floor_gib), float(pct) * total_gib)


def resolve_margins(
    device,
    margin_value,
    hard_value,
    *,
    margin_env: str,
    hard_env: str,
) -> WddmMargins:
    """Resolve the phase's margins from config value > env > auto.

    ``margin_value`` / ``hard_value`` are the config-supplied values (``None``
    means "consult the env var"; a negative margin or "auto" means auto).
    ``margin`` is clamped to at least ``hard``.
    """
    hard_gib = float(_env(hard_env, "1.0")) if hard_value is None else float(hard_value)
    raw = _env(margin_env, "-1.0") if margin_value is None else margin_value
    source = "env" if margin_value is None else "config"
    try:
        margin_gib = float(raw)
        auto = margin_gib < 0
    except (TypeError, ValueError):
        auto = str(raw).strip().lower() == "auto"
        margin_gib = -1.0
    if auto:
        margin_gib = auto_margin_gib(device)
        source = "auto"
    return WddmMargins(
        hard_gib=hard_gib,
        margin_gib=max(margin_gib, hard_gib or 0.0),
        source=source,
    )


def cap_fraction(total_bytes, free_bytes, reserved_bytes, hard_gib) -> float:
    """Allocator-cap fraction so device_used stays <= total - hard (pure).

    The cap governs torch's own reserved pool, but torch is not the card's
    only tenant (``non_torch``). Capping torch at ``total - hard`` alone lets
    device_used reach ``total - hard + non_torch`` (observed: reserved 10.97 +
    non_torch 1.02 = 11.99/11.99 GiB, device_free 0, silent WDDM paging).
    Subtract the measured non-torch share so the whole device, not just torch,
    keeps the hard margin free.
    """
    total = float(max(1, int(total_bytes)))
    non_torch = max(0.0, (total - float(free_bytes)) - float(reserved_bytes))
    cap_bytes = total - float(hard_gib) * GIB - non_torch
    return max(0.1, min(1.0, cap_bytes / total))


def sampling_allocator_budget_free_bytes(
    total_bytes,
    allocated_bytes,
    cap_fraction,
    hard_bytes,
    *,
    gc_threshold=0.95,
):
    """Allocated-side equivalent of driver-free for the sampling planner (pure).

    Driver-free counts torch's idle cached segments as *used*, so a plan built
    from it refuses residency that the allocator cap's GC would reclaim on
    demand. The allocator-side capacity is governed by the gc target
    (``gc_threshold * cap``): live allocations may safely grow to it, and the
    planner's margin beyond the hard floor (which is already inside the cap)
    stays free below the target as the fragmentation/allowance pad.

    Returned in the same units/meaning as ``mem_get_info`` free so the
    downstream ``usable = free - working_reserve - margin`` keeps its shape:

        usable = threshold*cap - allocated - working_reserve - (margin - hard)

    Returns ``None`` when no cap fraction is known (non-Windows / cap not
    applied); callers should then stay on the driver-free number.
    """
    if cap_fraction is None:
        return None
    cap_bytes = float(cap_fraction) * float(max(1, int(total_bytes)))
    return int(
        float(gc_threshold) * cap_bytes
        - float(max(0, int(allocated_bytes)))
        + float(max(0, int(hard_bytes)))
    )


def sampling_guard_predicted_peak_free(total_b, free_b, reserved_b, peak_reserved_b):
    """Predicted free VRAM at the next forward's peak (pure).

    ``non_torch = (total - free) - reserved`` plus the worst forward's reserved
    high-water is what the next peak will occupy; the prediction is ``total``
    minus that. It shrinks one-for-one as external use grows -- which is the
    cohabitation guard's trigger. Forward-only sampling never OOMs at the cliff
    (it pages silently), so the guard watches this instead of an exception.
    """
    other_b = max(0, (total_b - free_b) - reserved_b)
    return total_b - (peak_reserved_b + other_b)


def training_cliff_predicted_peak_free_gib(
    total_gib, device_free_gib, torch_reserved_gib, peak_allocated_gib
):
    """Driver free expected when the next step rebuilds its live peak (pure).

    ``empty_cache`` can make step-end free look healthy by dropping idle cached
    blocks, but the next forward/backward will recreate the live peak. Keep
    non-allocator residents (``non_torch``) from the current snapshot and ask
    whether peak allocated memory itself clears the WDDM hard floor.
    """
    device_used_gib = max(0.0, total_gib - device_free_gib)
    non_torch_gib = max(0.0, device_used_gib - torch_reserved_gib)
    return total_gib - (max(0.0, peak_allocated_gib) + non_torch_gib)


def training_promotion_worst_shape_free_gib(
    *,
    resident_gib,
    added_block_gib,
    ring_gib,
    worst_working_reserve_gib,
    other_gib,
    total_gib,
):
    """Predicted device-free margin on the WORST measured resolution after
    promoting one resident block (pure, CPU-testable).

    Residency is global -- a block promoted to resident stays resident for every
    resolution bucket -- but the activation working set is not: the largest
    measured resolution defines the tightest cohabitation peak. A promotion
    decided on a roomy low-res step still has to leave room for the worst measured
    resolution's working set, or that high-res step silently pages across the WDDM
    dedicated cliff (which no allocator retry counter catches). So gate the
    promotion on the worst measured working reserve, not the current step's free.

    ``worst_working_reserve_gib`` is the reserve the plan actually applies to every
    shape (already the max across measured buckets in the training controller).
    Conservative / from-below: assumes the promoted block adds its full size to the
    resident baseline and the ring does not shrink to compensate. Returns the
    predicted worst-case device-free GiB; the caller vetoes the promotion when it
    would fall below the promote floor.
    """
    predicted_used = (
        max(0.0, float(resident_gib))
        + max(0.0, float(added_block_gib))
        + max(0.0, float(ring_gib))
        + max(0.0, float(worst_working_reserve_gib))
        + max(0.0, float(other_gib))
    )
    return float(total_gib) - predicted_used


def training_eager_promote_blocks(
    *,
    resident_gib,
    block_gib,
    ring_gib,
    worst_working_reserve_gib,
    other_gib,
    total_gib,
    promote_floor_gib,
    max_blocks,
):
    """How many equal-sized blocks may be promoted at once while keeping the
    predicted worst-shape free margin at or above ``promote_floor_gib`` (pure).

    This is the eager-fill counterpart of the one-block-at-a-time climb: a roomy
    card leaves GiBs idle if residency only ever grows one block per cadence
    window. The prediction is the same conservative worst-measured-resolution
    model as ``training_promotion_worst_shape_free_gib`` -- the blocks are assumed
    to add their full size and the ring is assumed not to shrink -- so the floor is
    what the run actually keeps free on its tightest measured shape. Returns 0 when
    not even one block fits, which the caller reports as a worst-shape veto.
    """
    block = float(block_gib)
    limit = int(max_blocks)
    if block <= 0.0 or limit <= 0:
        return 0
    free_now = training_promotion_worst_shape_free_gib(
        resident_gib=resident_gib,
        added_block_gib=0.0,
        ring_gib=ring_gib,
        worst_working_reserve_gib=worst_working_reserve_gib,
        other_gib=other_gib,
        total_gib=total_gib,
    )
    room = free_now - float(promote_floor_gib)
    if room < block:
        return 0
    return min(limit, int(room // block))


def sampling_step_should_trim(free_before_b, trim_margin_b) -> bool:
    """Whether realized device-free warrants a per-step cache trim (pure).

    WDDM pages on the committed footprint silently, so the trigger is realized
    free, not an allocated-side or peak signal. Trim (empty_cache) is cheap and
    non-destructive, so the bar is just "free has dropped into the margin."
    """
    return free_before_b < trim_margin_b


def sampling_step_should_demote(free_after_b, hard_floor_b) -> bool:
    """Whether to escalate to a block demote after a trim (pure).

    Only when trimming left free still under the hard floor -- i.e. there was
    no idle cache to reclaim, so the pressure is real (external) and the only
    relief is giving back resident weights.
    """
    return free_after_b < hard_floor_b


def estimate_sampling_working_reserve_bytes(
    image_tokens: int,
    text_tokens: int = 512,
    *,
    batch_cfg: bool = False,
    fp8_native: bool = True,
    base_bytes: int = int(2.2 * GIB),
    per_token_bytes: int = 40 * 1024,
    dequant_pad_bytes: int = int(1.4 * GIB),
    safety: float = 1.15,
    headroom_bytes: int = GIB,
) -> int:
    """Cold-start estimate of the sampling working set (pure, CPU-testable).

    Used before any measured peak exists, so a high-resolution first sample
    plans enough streaming up front instead of discovering the working set
    via mid-denoise OOM demotions (every demote invalidates compiled state
    and, under strict ingraph, changes the pack set).

    Linear-in-tokens model calibrated on Krea2 RTX 4070 smoke runs
    (2026-07-07/08, fp8 + cutlass attention, sequential CFG, partial
    residency; ``sampling_extra`` = peak allocated minus resident weights):

        512x512  -> L = 1024 + 512 = 1536 tokens,  extra ~= 2.2 GiB
        2000x2000-> L = 15625 + 512 = 16137 tokens, extra ~= 2.8 GiB
        => per_token ~= 40 KiB, base ~= 2.2 GiB (streaming/dequant churn +
           fixed workspaces dominate; per-token activations are small)

        512x512 dequant fallback (fp8 sampling off) -> extra ~= 3.6 GiB
        => dequant_pad ~= 1.4 GiB (torchao fp32 dequant transients)

    Batched CFG scales the token-dependent share by 2.5, not 2.0: besides the
    batch-2 doubling, the fp32 intermediates (observed as (2, L, features)
    fp32 allocations, 740 MiB each at 2000px) are underweighted in the
    batch-1-calibrated per-token constant -- the x2.0 estimate ran ~1 GiB
    short at 2000px (two demote rounds). ``safety`` covers model-to-model
    variation, and
    ``headroom_bytes`` (flat +1 GiB) deliberately overestimates: streaming
    one extra block costs a little bandwidth, while underestimating costs a
    mid-denoise demote -- which invalidates compiled state, mutates the
    strict-ingraph pack set, and (observed at 2000px) can cascade into a
    full streamed transition. The learned per-run reserve replaces this
    estimate after the first measured sample.
    """
    tokens = max(0, int(image_tokens)) + max(0, int(text_tokens))
    token_bytes = int(tokens * per_token_bytes * (2.5 if batch_cfg else 1.0))
    estimate = int(base_bytes) + token_bytes
    if not fp8_native:
        estimate += int(dequant_pad_bytes)
    return int(estimate * float(safety)) + int(headroom_bytes)


def estimate_training_working_reserve_bytes(
    image_tokens: int,
    text_tokens: int = 512,
    *,
    base_bytes: int = int(5.2 * GIB),
    per_token_bytes: int = 610 * 1024,
    safety: float = 1.15,
    headroom_bytes: int = GIB,
) -> int:
    """Cold-start estimate of the training working set (pure, CPU-testable).

    Training's cold-start reserve was a flat constant (planner.py's
    ``DEFAULT_AUTO_WORKING_RESERVE_GIB = 5.0``) with no resolution awareness at
    all, unlike sampling's shape-aware ``estimate_sampling_working_reserve_bytes``
    above. At low resolution that flat reserve is generous; at high resolution
    it is not enough, so the attach-time residency plan keeps too many blocks
    resident, leaves activations too little headroom, and the run discovers the
    shortfall only via a cold-start WDDM-cap-violation storm -- each violation
    widens the allocator cap by a fixed ``WDDM_CAP_RELIEF_BYTES`` (0.5 GiB), so
    a large resolution jump can cost several wasted/skipped steps before the
    cap finally catches up (observed: Krea2 LoKr at 1024x1024 skipped 5/5 fake
    steps under the flat default, never reaching a real step).

    Linear-in-tokens model calibrated on Krea2 LoKr RTX 4070 smoke runs
    (2026-07-14, ``--block-stream-only`` so zero blocks are resident and
    ``torch_max_allocated`` is purely the forward+backward+optimizer
    footprint, uncontaminated by the residency split this estimate feeds):

        512x512   -> 1024 tokens, torch_max_allocated ~= 5.76 GiB
        1024x1024 -> 4096 tokens, torch_max_allocated ~= 7.50 GiB
        => per_token ~= 595 KiB, base ~= 5.17 GiB (rounded to 610 KiB / 5.2 GiB)

    Only two points, one adapter variant, one card -- weaker calibration than
    the sampling estimator above. ``text_tokens`` is folded in at the same
    per-token rate by symmetry with the sampling model; it was held constant
    across both calibration runs, not independently measured. ``safety`` and
    the flat ``headroom_bytes`` deliberately overestimate: streaming one extra
    block is cheap, under-reserving costs the cap-violation storm above.
    """
    tokens = max(0, int(image_tokens)) + max(0, int(text_tokens))
    estimate = int(base_bytes) + int(tokens * per_token_bytes)
    return int(estimate * float(safety)) + int(headroom_bytes)


def sampling_overshoot_margin_bytes(
    overshoot_gib: float = 0.86,
    safety_gib: float = 0.375,
    hard_bytes: int = 0,
) -> int:
    """Auto sampling margin in the allocator-cap era (pure, CPU-testable).

    With the reclaim allocator cap guarding the WDDM cliff (a capped allocation
    recycles cache or raises a loud OOM, it never silently pages), the sampling
    margin's only remaining job is to cover the caching allocator's
    reserved-over-allocated overshoot. Measured on Krea2 fp8 512px that overshoot
    is ~0.86 GiB and rock-steady (std ~0.05 GiB across 8 seeds), so the auto
    margin is that measured overshoot plus one safety block -- NOT the old
    ``0.10 * card`` cushion, which was sized for a chaotic allocator that kept
    jumping over the limit and no longer misbehaves. Narrowing it hands the
    difference straight to resident weights (fewer streamed blocks). Floored at
    the hard margin so it can never drop below the WDDM device-free floor.
    """
    return int(
        max(float(overshoot_gib) + float(safety_gib), float(max(0, int(hard_bytes))) / GIB)
        * GIB
    )


def training_guard_pressure(dxgi: dict, physical: dict) -> dict:
    """Combine the DXGI LOCAL and physical cliff signals (pure).

    Pressure if EITHER signal predicts the next step's peak crosses its floor.
    The DXGI LOCAL budget is a per-process OS grant and its usage counter
    excludes other processes, so it can bless a layout the physical
    (mem_get_info) view already knows will overfill the card -- and vice versa
    when the OS shrinks the budget early. The merged dict keeps the DXGI
    fields at the top level (``source`` compatibility) and carries the
    physical signal under ``physical_*``.
    """
    merged = dict(dxgi)
    merged["pressure"] = bool(dxgi.get("pressure")) or bool(physical.get("pressure"))
    merged["physical_predicted_peak_free_gib"] = physical.get("predicted_peak_free_gib")
    merged["physical_target_free_gib"] = physical.get("target_free_gib")
    merged["pressure_sources"] = [
        src["source"] for src in (dxgi, physical) if src.get("pressure")
    ]
    return merged


# ---------------------------------------------------------------------------
# Two-timescale residency control (see tasks/done/RESIDENCY_TWO_TIMESCALE_PLAN.md)
#
# Allowance lives in *target-space* (0.95*cap - live); the allocator cap is set
# in *cap-space*. The two differ by the gc_threshold factor: a cap raise of ``d``
# only adds ``gc_threshold * d`` of GC target / allowance. Every conversion below
# carries the ``/ gc_threshold`` so no call site open-codes it (that missing
# divisor silently under-reserves and licenses a promotion that immediately binds).
#
# All functions here are pure/CPU-testable. The cap and residency levers both act
# only at phase boundaries, and a cap change is realized lazily -- on the next
# fresh cudaMalloc, i.e. the next forward()/step -- so the controller reads
# counters that lag its move by one window (the FSM's verify phases absorb this).
# ---------------------------------------------------------------------------

GC_THRESHOLD = 0.95


def allocator_allowance_bytes(cap_bytes, live_bytes, *, gc_threshold=GC_THRESHOLD) -> int:
    """Idle-cache allowance under the cap: ``gc_threshold*cap - live`` (pure).

    The caching allocator sweeps idle segments when reserved would cross the GC
    target ``gc_threshold * cap``; live bytes (residents + ring + activations)
    count against that target but cannot be freed. So the room left for reusable
    idle cache -- the buffer between smooth reuse and a fresh-cudaMalloc sweep --
    is ``gc_threshold*cap - live``. Negative means live alone exceeds the target:
    every sweep dumps all cache and every reuse re-mallocs (self-sustaining
    thrash), so callers must keep this positive at the live peak.
    """
    return int(float(gc_threshold) * float(max(0, int(cap_bytes))) - float(max(0, int(live_bytes))))


def cap_bytes_for_live(
    planned_live_bytes,
    cache_budget_bytes,
    cliff_cap_bytes,
    *,
    floor_cap_bytes=0,
    gc_threshold=GC_THRESHOLD,
) -> int:
    """Cap that hosts ``planned_live`` plus an idle-cache budget (pure).

    Inverse of :func:`allocator_allowance_bytes`: to let live grow to
    ``planned_live`` while keeping ``cache_budget`` of reusable idle cache under
    the GC target, the cap must be ``(planned_live + cache_budget) / gc_threshold``.
    Clamped to the WDDM cliff bound above (never license silent paging; see
    :func:`cap_fraction`) and an optional floor below.
    """
    want = (float(max(0, int(planned_live_bytes))) + float(max(0, int(cache_budget_bytes)))) / float(gc_threshold)
    want = min(want, float(int(cliff_cap_bytes)))
    want = max(want, float(max(0, int(floor_cap_bytes))))
    return int(want)


def cap_can_host_promotion(
    live_bytes,
    block_bytes,
    slack_pad_bytes,
    cliff_cap_bytes,
    *,
    gc_threshold=GC_THRESHOLD,
) -> bool:
    """Can the cheap cap lever (tier 1) absorb one more resident block? (pure).

    Promoting a streamed block to resident raises live by ``block_bytes``. To
    keep ``slack_pad_bytes`` of allowance afterward, the GC target must reach
    ``live + block + slack``, i.e. the cap must reach
    ``(live + block + slack) / gc_threshold``. The cap lever can do this only if
    that target cap is still under the WDDM cliff bound; otherwise the cap is
    pinned at the cliff and the allowance must come from lowering live -- an
    expensive resident demote (tier 2).

    The ``/ gc_threshold`` is load-bearing: a naive ``cliff - cap >= block`` test
    under-reserves by the 0.95 factor.
    """
    need_cap = (
        float(max(0, int(live_bytes)))
        + float(max(0, int(block_bytes)))
        + float(max(0, int(slack_pad_bytes)))
    ) / float(gc_threshold)
    return need_cap <= float(int(cliff_cap_bytes))


def residency_promote_ok(
    num_alloc_retries,
    allocator_slack_bytes,
    block_bytes,
    slack_pad_bytes,
) -> bool:
    """Sampling climb gate: convert one streamed block to resident? (pure).

    Approach residency from below (undershoot-and-climb): only add a block when
    the telemetry proves the room is really there --

      * ``num_alloc_retries == 0`` over the window (nothing cap-binding), AND
      * worst-shape allocator slack (``0.95 * cap - predicted_live``)
        exceeds one block plus the pad, so the promotion still leaves
        ``slack_pad`` of reusable-cache allowance.

    Both must hold: retries can be zero simply because residency is too low, so
    the worst-shape allocator-slack test is what proves there is room to spend.
    """
    if int(num_alloc_retries or 0) > 0:
        return False
    return float(allocator_slack_bytes or 0) > float(block_bytes) + float(slack_pad_bytes)


# --- Hysteresis FSM (one transition per phase boundary) ---------------------
#
# States mirror the plan's state machine. DEMOTE_REQUIRED is folded into the
# transition (emit "demote" and land in COLD) since the ring resize is a
# synchronous boundary transaction, not a state that waits a window.

FSM_COLD = "cold"                          # measurements invalid (post-compile/retrace/demote)
FSM_STABLE = "stable"                      # clean + eligible for the from-below climb
FSM_CAP_VERIFY = "cap_verify"              # cap raised, confirming it took
FSM_PROMOTION_VERIFY = "promotion_verify"  # one block promoted, confirming clean
FSM_COOLDOWN = "cooldown"                  # re-promotion barred N windows after a rollback

ACT_HOLD = "hold"
ACT_RAISE_CAP = "raise_cap"
ACT_PROMOTE = "promote"
ACT_DEMOTE = "demote"
ACT_ROLLBACK = "rollback"


@dataclass(frozen=True)
class ResidencyFsmState:
    name: str = FSM_COLD
    windows_in_state: int = 0


def residency_fsm_step(
    state: ResidencyFsmState,
    signals: dict,
    *,
    k_clean: int = 2,
    k_verify: int = 2,
    cooldown_n: int = 4,
) -> tuple[ResidencyFsmState, str]:
    """Advance the residency controller one phase boundary (pure, CPU-testable).

    ``signals`` (all read as bools unless noted):
      * ``measurements_invalid`` -- a recompile / retrace / layout change happened;
        every counter read across it is meaningless -> force COLD.
      * ``binding`` -- retries or external pressure this window (allowance too low).
      * ``cap_can_relieve`` -- a cap raise can restore the pad at current live
        (cliff has room); if False under pressure, only a demote can.
      * ``promote_gate`` -- :func:`residency_promote_ok` verdict (there is slack).
      * ``cap_covers_promo`` -- the cliff already hosts the promotion with no raise
        (:func:`cap_can_host_promotion` at the current cap headroom).

    Returns ``(next_state, action)`` with ``action`` in the ``ACT_*`` set. The
    verify phases each span ``k_verify`` windows because a cap/residency move only
    shows its signal on the *next* step (the one-window GC lag).
    """
    name = state.name
    w = state.windows_in_state + 1

    invalid = bool(signals.get("measurements_invalid"))
    binding = bool(signals.get("binding"))
    cap_relieve = bool(signals.get("cap_can_relieve"))
    promote_gate = bool(signals.get("promote_gate"))
    cap_covers = bool(signals.get("cap_covers_promo"))

    def stay(action=ACT_HOLD):
        return ResidencyFsmState(name, w), action

    def enter(new_name, action=ACT_HOLD):
        return ResidencyFsmState(new_name, 0), action

    # A demote is a synchronous transaction that invalidates the layout -> COLD.
    def demote():
        return ResidencyFsmState(FSM_COLD, 0), ACT_DEMOTE

    if invalid and name not in (FSM_COLD,):
        return enter(FSM_COLD)

    if name == FSM_COLD:
        if invalid or binding:
            return ResidencyFsmState(FSM_COLD, 0 if invalid else w), ACT_HOLD
        return enter(FSM_STABLE) if w >= k_clean else stay()

    if name == FSM_STABLE:
        if binding:
            return enter(FSM_CAP_VERIFY, ACT_RAISE_CAP) if cap_relieve else demote()
        # Eligible to climb only once the state has held clean for k_clean windows.
        if promote_gate and w >= k_clean:
            if cap_covers:
                return enter(FSM_PROMOTION_VERIFY, ACT_PROMOTE)
            return enter(FSM_CAP_VERIFY, ACT_RAISE_CAP)  # pre-fund, promote after verify
        return stay()

    if name == FSM_CAP_VERIFY:
        if binding:
            return demote()  # the raise didn't relieve -> shed live
        return enter(FSM_STABLE) if w >= k_verify else stay()

    if name == FSM_PROMOTION_VERIFY:
        if binding:
            return enter(FSM_COOLDOWN, ACT_ROLLBACK)
        if w == 1:
            return stay()  # ignore only layout-contaminated non-binding signals
        return enter(FSM_STABLE) if w >= k_verify + 1 else stay()

    if name == FSM_COOLDOWN:
        if binding:
            return demote()  # demote still allowed during cooldown
        return enter(FSM_STABLE) if w >= cooldown_n else stay()

    # Unknown state: fail safe to COLD.
    return enter(FSM_COLD)
