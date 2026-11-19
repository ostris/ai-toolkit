"""Beats, bar heads and meters from a bar head's per-frame probabilities: one Viterbi pass of an HMM over
(beats in the bar B, beat k of the bar, beat interval, position within the beat) at 100 fps — madmom's bar
pointer written out, with several meters at once.

* The beat interval changes only at a beat, by ``exp(-lam |new / old - 1|)`` (ritardando, accelerando).
* At a bar line the meter may change (``meter_change``), and the beat unit may halve or double (``switch``:
  4/4 to 6/8 counted in eighths and back).
* Observations: the beat probability within +-``HALF`` frames of a beat (the bar-head probability on k = 0),
  and at each beat the head's "beats in this bar" and "beats left to the next bar head" (weight ``w_bar``).
* Candidate meters: those the head gives ``metre_min`` of the song on average, its favourite, and any it holds
  for ``local_seconds`` somewhere (mean probability ``local_min``), so a two-bar 3/4 intro still counts.
* ``p_hold`` > 0 adds a hold state per row and interval, for fermatas (off by default).
* ``denominators`` writes each tracked bar's denominator from a head's ``bar_unit``, keeping the written tempo.
* ``refold`` reads runs of bars at double tempo inside the same meter again at the surrounding tempo (a head that
  hears a bar's eighths as beats, or an inserted 2/4 bar the head does not count as two beats).

Built for ``heads.FrameHead`` outputs at 25 Hz (``beats.HeadBeats``); numba compiles ``viterbi`` on first use
(cached on disk).
"""

from __future__ import annotations

import numpy as np
from numba import njit

FPS, HEAD_RATE, HALF = 100, 25, 4
BAND = 200.0  # transitions further than this below a target's best source are dropped (below e^-200)


def _norm_rows(x: np.ndarray) -> np.ndarray:
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


@njit(cache=True)
def viterbi(taus, starts, size, inner, first, last, k0, to_down, is_first, beat_src, beat_val, bar_src, bar_val, metre_tr,
            gain_down, gain_beat, bar_ev, half, p_hold, q_hold):  # fmt: skip
    """The HMM's best path, as (beat frames, bar-head frames), last first.  Every (row, interval) keeps its
    positions in a ring: position p at frame t sits in slot (t - p) % tau, so nothing moves from frame to frame;
    the slot t % tau holds the beat that ends at t - 1 and then the one that starts at t."""
    frames, nbk, nt, nb = len(gain_down), bar_ev.shape[1], len(taus), len(last)
    buf = np.full((nbk, size), -np.log(nbk * size))
    back = np.zeros((frames, nbk, nt), np.int32)
    end = np.empty((nbk, nt))
    start = np.empty((nbk, nt))
    tbest = np.empty((nb, nt))
    targ = np.zeros((nb, nt), np.int64)
    # a fermata: after a beat's full length the bar may hold (one state per row and interval, staying with q_hold
    # per frame) before the next beat comes at the same interval; the next beat then starts from the hold
    hold = np.full((nbk, nt), -np.inf)
    stay = np.zeros((frames, nbk, nt), np.bool_)
    via = np.zeros((nbk, nt), np.bool_)
    off = nbk * nt
    use_hold = p_hold > 0
    l_go, l_enter = np.log(1 - p_hold), np.log(p_hold) if use_hold else -np.inf
    l_stay, l_exit = (np.log(q_hold), np.log(1 - q_hold)) if use_hold else (-np.inf, -np.inf)
    for t in range(frames):
        for r in range(nbk):
            for j in range(nt):
                e = buf[r, starts[j] + t % taus[j]]
                if use_hold:
                    d, h = e + l_go, hold[r, j] + l_exit
                    via[r, j] = h > d
                    end[r, j] = h if h > d else d
                    a, b = hold[r, j] + l_stay, e + l_enter
                    stay[t, r, j] = a >= b
                    hold[r, j] = a if a >= b else b
                else:
                    end[r, j] = e
        for r in inner:  # the next beat of the same bar
            for j in range(nt):
                best, arg = -np.inf, 0
                for w in range(beat_src.shape[1]):
                    v = end[r - 1, beat_src[j, w]] + beat_val[j, w]
                    if v > best:
                        best, arg = v, beat_src[j, w]
                start[r, j] = best
                back[t, r, j] = (r - 1) * nt + arg + (off if via[r - 1, arg] else 0)
        for m in range(nb):  # a bar ends
            for j in range(nt):
                best, arg = -np.inf, 0
                for w in range(bar_src.shape[1]):
                    v = end[last[m], bar_src[j, w]] + bar_val[j, w]
                    if v > best:
                        best, arg = v, bar_src[j, w]
                tbest[m, j], targ[m, j] = best, arg
        for m2 in range(nb):  # the next bar, of any metre
            for j in range(nt):
                best, arg = -np.inf, 0
                for m in range(nb):
                    v = tbest[m, j] + metre_tr[m, m2]
                    if v > best:
                        best, arg = v, m
                start[first[m2], j] = best
                back[t, first[m2], j] = last[arg] * nt + targ[arg, j] + (off if via[last[arg], targ[arg, j]] else 0)
        for r in range(nbk):
            for j in range(nt):
                tau, base = taus[j], starts[j]
                buf[r, base + t % tau] = start[r, j] + bar_ev[t, r]
                g = gain_down[t] if k0[r] else gain_beat[t]
                for p in range(half + 1):  # just after the beat
                    buf[r, base + (t - p) % tau] += g
                g = gain_down[t] if to_down[r] else gain_beat[t]
                for p in range(tau - half, tau):  # just before the next one
                    buf[r, base + (t - p) % tau] += g
        if t % 256 == 255:  # renormalise (the hold scores too: they are on the same scale)
            top = buf.max()
            buf -= top
            hold -= top
    # the best final state, then back along the stored choices
    best, row, j, p = -np.inf, 0, 0, 0
    for r in range(nbk):
        for jj in range(nt):
            for slot in range(taus[jj]):
                v = buf[r, starts[jj] + slot]
                if v > best:
                    best, row, j, p = v, r, jj, (frames - 1 - slot) % taus[jj]
    beats, downs = [], []
    t = frames - 1
    while True:
        t0 = t - p
        if t0 < 0:
            break
        beats.append(t0)
        if is_first[row]:
            downs.append(t0)
        prev = back[t0, row, j]
        held = prev >= off
        prev = prev % off
        row, j = prev // nt, prev % nt
        t = t0 - 1
        if held:  # back over the hold to the frame it was entered at; the beat before ended the frame before
            while t > 0 and stay[t, row, j]:
                t -= 1
            t -= 1
        p = taus[j] - 1
    return beats, downs


def candidate_metres(bar_beats: np.ndarray, metre_min: float = 0.04, local_min: float = 0.5,
                     local_frames: int = 200) -> list[int]:  # fmt: skip
    """Beats-per-bar the tracker considers: ``bar_beats`` [frames, 16] (probabilities of 1..16 beats)."""
    mean = bar_beats.mean(0)
    metres = {b + 1 for b in range(1, 16) if mean[b] >= metre_min} | {int(mean[1:].argmax()) + 2}
    if local_min and len(bar_beats) > local_frames:
        c = np.cumsum(np.vstack([np.zeros((1, bar_beats.shape[1])), bar_beats]), axis=0)
        peak = ((c[local_frames:] - c[:-local_frames]) / local_frames).max(0)
        metres |= {b + 1 for b in range(1, 16) if peak[b] >= local_min}
    return sorted(metres)


def track(pb: np.ndarray, pd: np.ndarray, bar_beats: np.ndarray, beats_left: np.ndarray, lam: float = 100.0,
          switch: float = 0.02, meter_change: float = 0.02, w_bar: float = 1.0, w_down: float = 1.0,
          min_bpm: float = 50.0, max_bpm: float = 440.0, n_tau: int = 60, metre_min: float = 0.04,
          local_min: float = 0.5, local_seconds: float = 2.0, p_hold: float = 0.0,
          q_hold: float = 0.98) -> tuple[np.ndarray, np.ndarray, list[int]]:  # fmt: skip
    """Beat and bar-head times (seconds) and the candidate meters, from per-frame probabilities at ``FPS``:
    beat ``pb`` and bar head ``pd`` [frames], ``bar_beats`` and ``beats_left`` [frames, 16] (classes 1..16)."""
    taus = np.unique(np.round(np.geomspace(60 * FPS / max_bpm, 60 * FPS / min_bpm, n_tau)).astype(int))
    nt = len(taus)
    starts = np.concatenate([[0], np.cumsum(taus)[:-1]])
    size = int(taus.sum())
    metres = candidate_metres(bar_beats, metre_min, local_min, int(local_seconds * FPS))
    rows = [(b, k) for b in metres for k in range(b)]
    nbk = len(rows)
    k0 = np.array([k == 0 for _, k in rows])
    to_down = np.array([k == b - 1 for b, k in rows])  # the next beat is a bar head
    last = np.array([i for i, (b, k) in enumerate(rows) if k == b - 1])  # one per metre, in metre order
    first = np.array([i for i, (b, k) in enumerate(rows) if k == 0])
    inner = np.array([i for i, (b, k) in enumerate(rows) if k > 0])  # their predecessor is i - 1
    nb = len(metres)
    r = taus[None, :] / taus[:, None]  # [from, to]
    beat_tr = _norm_rows(-lam * np.abs(r - 1))
    bar_tr = _norm_rows(np.logaddexp.reduce(np.stack([-lam * np.abs(r - 1), np.log(switch) - lam * np.abs(2 * r - 1),
                                                      np.log(switch) - lam * np.abs(r / 2 - 1)]), axis=0))  # fmt: skip
    metre_tr = np.full((nb, nb), np.log(meter_change / max(1, nb - 1)))
    np.fill_diagonal(metre_tr, np.log(1 - meter_change) if nb > 1 else 0.0)
    eps = 1e-6
    pb, pd = np.clip(pb, eps, 1 - eps), np.clip(pd, eps, 1 - eps)
    # a fixed +-HALF beat region at every tempo (madmom's 1 / (obs_lambda - 1) share would put beats on weak
    # off-beat peaks); every frame outside the regions gets log(1 - pb) in every state, so only the regions need
    # an observation, relative to it
    l_other = np.log(1 - pb)
    gain_down = np.log(pb) + w_down * np.log(pd) - l_other
    gain_beat = np.log(pb) + w_down * np.log(1 - pd) - l_other
    bar_ev = np.zeros((len(pb), nbk))  # evidence added when a beat starts
    lbb, lbl = np.log(np.clip(bar_beats, eps, 1)), np.log(np.clip(beats_left, eps, 1))
    for i, (b, k) in enumerate(rows):
        bar_ev[:, i] = w_bar * lbl[:, min(b - k, 16) - 1] + (w_bar * lbb[:, min(b, 16) - 1] if k == 0 else 0.0)

    def band(tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # per target: the sources within BAND, padded
        keep = [np.flatnonzero(tr[:, j] >= tr[:, j].max() - BAND) for j in range(nt)]
        width = max(len(k) for k in keep)
        src = np.array([np.pad(k, (0, width - len(k)), constant_values=k[0]) for k in keep])  # [to, width]
        val = np.array([np.pad(tr[k, j], (0, width - len(k)), constant_values=-np.inf) for j, k in enumerate(keep)])
        return src, val

    beat_src, beat_val = band(beat_tr)
    bar_src, bar_val = band(bar_tr)
    is_first = np.zeros(nbk, np.bool_)
    is_first[first] = True
    i64 = lambda a: np.asarray(a).astype(np.int64)  # noqa: E731
    beats, downs = viterbi(i64(taus), i64(starts), size, i64(inner), i64(first), i64(last), k0, to_down, is_first,
                           i64(beat_src), beat_val, i64(bar_src), bar_val, metre_tr, gain_down, gain_beat, bar_ev, HALF,
                           p_hold, q_hold)  # fmt: skip
    return np.asarray(sorted(beats)) / FPS, np.asarray(sorted(downs)) / FPS, metres


def track_head(probs: dict[str, np.ndarray], **options) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """``track`` on a head's 25 Hz probabilities (``beat``, ``downbeat`` [frames]; ``bar_beats``, ``beats_left``
    [frames, 16]), the binary ones interpolated to 100 fps and the bar facts held per head frame."""
    head_t = (np.arange(len(probs["beat"])) + 0.5) / HEAD_RATE
    t100 = np.arange(int(head_t[-1] * FPS) + 1) / FPS
    near = np.minimum((t100 * HEAD_RATE).astype(int), len(head_t) - 1)
    return track(np.interp(t100, head_t, probs["beat"]), np.interp(t100, head_t, probs["downbeat"]),
                 probs["bar_beats"][near], probs["beats_left"][near], **options)  # fmt: skip


def apply_offsets(beats: np.ndarray, downbeats: np.ndarray, beat_prob: np.ndarray, beat_offset: np.ndarray,
                  limit: float = 0.06) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Beat times moved by a head's sub-frame offsets (``heads.OFFSETS``): each beat goes to the likeliest beat
    frame within one frame of it plus that frame's ``beat_offset`` (frames from its centre), unless that moves it
    more than ``limit`` seconds; bar heads move with their beat."""
    new = np.asarray(beats, dtype=np.float64).copy()
    for i, t in enumerate(new):
        j0 = int(t * HEAD_RATE)
        cand = [j for j in (j0 - 1, j0, j0 + 1) if 0 <= j < len(beat_prob)]
        if not cand:
            continue
        j = max(cand, key=lambda k: beat_prob[k])
        moved = (j + 0.5 + float(np.clip(beat_offset[j], -1.5, 1.5))) / HEAD_RATE
        if abs(moved - t) <= limit:
            new[i] = moved
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if len(new) and len(downbeats):
        downbeats = new[np.abs(np.asarray(beats)[None] - downbeats[:, None]).argmin(1)]
    return new, downbeats


def refold(beats: np.ndarray, downbeats: np.ndarray, low: float = 0.42, high: float = 0.58,
           context: int = 32) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Beats and bar heads with runs of half-length bars read at the surrounding tempo.  A bar is half-length when
    it lasts ``low``-``high`` times the median of the bars with as many beats within ``context`` bars either side.  A
    run of them with an even beat count, between full-length bars of that count, loses every other beat; its half
    bars are merged in pairs, and an odd one left over stays a half bar (a 2/4 bar in 4/4).  The bars before and after
    keep their bar lines, which a decode that forbade the double tempo would shift by half a bar from there on."""

    beats, downbeats = np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)
    bars = len(downbeats)
    if bars < 3:
        return beats, downbeats
    edges = np.append(downbeats, beats[-1] + 1e-3)
    inside = [np.flatnonzero((beats >= a - 1e-6) & (beats < b - 1e-6)) for a, b in zip(edges[:-1], edges[1:])]
    count = np.array([len(i) for i in inside])
    length = np.diff(edges)
    half = np.zeros(bars, dtype=bool)
    for i in range(bars - 1):  # the last bar's length is not known
        same = [j for j in range(max(0, i - context), min(bars - 1, i + context + 1)) if j != i and count[j] == count[i]]
        if same:
            half[i] = low < length[i] / np.median(length[same]) < high
    keep, merged = np.ones(len(beats), dtype=bool), np.zeros(bars, dtype=bool)
    i = 0
    while i < bars - 1:
        if not half[i]:
            i += 1
            continue
        j = i
        while j < bars - 1 and half[j] and count[j] == count[i]:
            j += 1
        if 0 < i and not half[i - 1] and count[i - 1] == count[i] == count[j] and not half[j] and count[i] % 2 == 0:
            for k in range(i, j):
                keep[inside[k][1::2]] = False
            merged[i + 1 : j : 2] = True
        i = j
    return beats[keep], downbeats[~merged]


def denominators(beats: np.ndarray, downbeats: np.ndarray, bar_unit: np.ndarray, duration: float,
                 units: tuple[int, ...] = (2, 4, 8, 16), tempo_weight: float = 4.0) -> list[int]:  # fmt: skip
    """Each tracked bar's denominator from a head's ``bar_unit`` probabilities [frames, len(units)] at 25 Hz,
    chosen so the written tempo holds: a Viterbi over the bars with the head's log-probability (mean over the bar)
    as each bar's score and ``tempo_weight * |ln(q / q_prev)|`` between neighbours, q = beat interval * unit / 4
    (the quarter note's length).  The same beat keeps its denominator (8 <-> 4 would halve or double q), and
    4/4 -> 7/8 at twice the beat speed is written /8 (7/4 would halve q); long stretches follow the head."""
    beats, downbeats = np.asarray(beats), np.asarray(downbeats)
    if not len(downbeats):
        return []
    ends = list(downbeats[1:]) + [duration]
    emit, ibi = [], []
    for start, end in zip(downbeats, ends):
        f0 = int(start * HEAD_RATE)
        p = bar_unit[f0 : max(int(end * HEAD_RATE), f0 + 1)].mean(0) if f0 < len(bar_unit) else np.full(len(units), 1 / len(units))
        emit.append(np.log(p + 1e-6))
        inside = beats[(beats >= start - 0.01) & (beats < end + 0.01)]
        ibi.append(float(np.median(np.diff(inside))) if len(inside) > 1 else np.nan)
    q = np.log(np.asarray(units, dtype=np.float64) / 4)
    score, back = emit[0].copy(), np.zeros((len(emit), len(units)), dtype=np.int64)
    for i in range(1, len(emit)):
        if np.isnan(ibi[i - 1]) or np.isnan(ibi[i]):
            cost = np.zeros((len(units), len(units)))
        else:
            cost = tempo_weight * np.abs((np.log(ibi[i]) + q)[None, :] - (np.log(ibi[i - 1]) + q)[:, None])
        total = score[:, None] - cost
        back[i] = total.argmax(0)
        score = total.max(0) + emit[i]
    path = [int(score.argmax())]
    for i in range(len(emit) - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return [units[j] for j in path[::-1]]
