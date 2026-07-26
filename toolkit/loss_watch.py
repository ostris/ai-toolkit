"""Per-image adaptive LR: online per-image loss tracking with an adaptive per-item loss
multiplier. Detection + multiplier only. No auto-recaption,
no exclusion — those are a separate follow-up.

Model- and network-agnostic by construction: it only ever sees (item_key, epoch, timestep, loss)
tuples handed to it from SDTrainer.calculate_loss(), which is the single shared loss path for
every architecture and both LoKr and LoRA. It never touches gradients, the network, or the
optimizer directly — its only output is a per-item float multiplier that gets written onto each
dataset item's existing `loss_multiplier` field, which SDTrainer already applies to the per-sample
loss before the batch mean. That plumbing was already there for static per-dataset weighting;
this just drives it dynamically.

Why timestep-normalize: a diffusion step's raw loss is dominated by the timestep drawn that step
(loss near t=1 is structurally larger than near t=0), so ranking images by raw loss mostly ranks
the dice roll, not the image. Every residual below is loss minus the current per-timestep-bucket
mean over the run so far.
"""
import os

_N_BUCKETS = 40  # timestep buckets over [0, 1] for the running-mean normalization


class PerImageAdaptiveLR:
    """Online per-image difficulty watcher that emits a loss multiplier per image.

    NOTE ON TERMINOLOGY: ai-toolkit has no epoch concept — runs are defined purely by a step
    count, and a dataset can be larger or smaller than that budget. So the `epoch` argument
    everywhere below is really a "window index": the caller picks a step-count window
    (BaseSDTrainProcess computes it as ~one pass over the dataset) and increments this index
    every time that many steps have elapsed. Internally it's just a monotonic counter that the
    trend/warmup/escalation logic counts against — it doesn't need to mean one literal dataset
    pass, though a window sized that way behaves closest to a literal per-epoch cadence.

    At each window boundary, classifies every image seen this run as STUCK (high residual, not
    descending — likely a bad caption or outlier), SUSPECT (early, extreme-magnitude outlier
    before a trend exists), EXHAUSTED (proved a good run, then plateaued — mined out), or healthy,
    and updates that image's multiplier accordingly:
      - confirmed stuck: throttled, escalating from `throttle_mult` toward `stuck_floor` the
        longer it stays confirmed
      - early suspect: mild throttle (`suspect_mult`) pending trend confirmation
      - exhausted: mild throttle (`exhausted_mult`) to curb overbake pressure
      - consistently healthy (from `easy_from_epoch` on): small boost (`easy_mult`) — the ONLY
        multiplier that goes above 1.0
      - everything else: 1.0

    No action during `warmup_epochs` (the trend needs data first). A verdict needs `persist_on`
    consecutive votes to be confirmed and `persist_off` consecutive clear votes to be released,
    so one noisy epoch can't flip it.
    """

    def __init__(
        self,
        *,
        warmup_epochs: int = 2,
        trend_window: int = 8,
        throttle_mult: float = 0.5,
        stuck_floor: float = 0.1,
        escalate_every: int = 2,
        suspect_mult: float = 0.7,
        exhausted_mult: float = 0.6,
        exhaust_drop_frac: float = 0.3,
        exhaust_on: int = 2,
        easy_mult: float = 1.1,
        easy_from_epoch: int = 3,
        healthy_min: int = 3,
        hi_q: float = 0.66,
        lo_q: float = 0.33,
        persist_on: int = 2,
        persist_off: int = 3,
        improve_frac: float = 0.12,
        improve_floor: float = 0.02,
        log_fn=None,
    ):
        self.warmup_epochs = warmup_epochs
        self.trend_window = trend_window  # epochs; split into two halves for the improve test
        self.throttle_mult = throttle_mult
        self.stuck_floor = stuck_floor
        self.escalate_every = escalate_every
        self.suspect_mult = suspect_mult
        self.exhausted_mult = exhausted_mult
        self.exhaust_drop_frac = exhaust_drop_frac
        self.exhaust_on = exhaust_on
        self.easy_mult = easy_mult
        self.easy_from_epoch = easy_from_epoch
        self.healthy_min = healthy_min
        self.hi_q = hi_q
        self.lo_q = lo_q
        self.persist_on = persist_on
        self.persist_off = persist_off
        self.improve_frac = improve_frac
        self.improve_floor = improve_floor
        # This codebase doesn't call logging.basicConfig() anywhere in the training path, so a
        # bare logging.getLogger() here would silently swallow anything below WARNING (the
        # "no longer stuck" / "restored state" messages) and print the rest inconsistently with
        # everything else the trainer outputs. Route through the caller's actual output channel
        # instead (BaseSDTrainProcess passes print_acc) — plain print() otherwise.
        self._log = log_fn or print

        self._records: list[tuple[str, int, int, float]] = []  # (key, epoch, bucket, loss)
        self._bsum = [0.0] * _N_BUCKETS
        self._bcnt = [0] * _N_BUCKETS

        self._mult: dict[str, float] = {}
        self.verdicts: dict[str, str] = {}

        self._stuck_votes: dict[str, int] = {}
        self._clear_votes: dict[str, int] = {}
        self._suspect_votes: dict[str, int] = {}
        self._exhaust_votes: dict[str, int] = {}
        self._stuck_epochs: dict[str, int] = {}  # consecutive epochs confirmed -> escalation
        self._confirmed_stuck: set[str] = set()
        self._last_reported_stuck: set[str] = set()
        self._healthy_epochs: dict[str, int] = {}
        self._retired: set[str] = set()  # proved a good run once; suppresses stuck/suspect after
        self._restored_residuals: dict[str, dict[int, float]] = {}  # from a prior run's checkpoint

    # ---- per-step ------------------------------------------------------------

    @staticmethod
    def _bucket(t: float) -> int:
        b = int(min(max(t, 0.0), 0.999999) * _N_BUCKETS)
        return min(b, _N_BUCKETS - 1)

    def multiplier(self, item_key: str) -> float:
        """Current loss multiplier for an item (looked up before the loss is scaled)."""
        return self._mult.get(str(item_key), 1.0)

    def observe(self, *, epoch: int, item_key: str, timestep: float, loss: float) -> None:
        """Record one image's loss for this step. Never raises into the training loop."""
        try:
            key = str(item_key)
            t = float(timestep)
            loss = float(loss)
            b = self._bucket(t)
            self._bsum[b] += loss
            self._bcnt[b] += 1
            self._records.append((key, int(epoch), b, loss))
        except Exception as e:
            self._log(f"[adaptive-lr] observe failed ({e})")

    # ---- epoch boundary --------------------------------------------------------

    def epoch_boundary(self, epoch: int) -> dict[str, str]:
        """Reclassify every image seen so far and refresh self._mult. Call once per epoch, after
        the last step of that epoch has been observe()'d. Returns the verdicts dict."""
        try:
            if epoch <= self.warmup_epochs or (not self._records and not self._restored_residuals):
                return self.verdicts

            # Per-timestep-bucket means over the WHOLE run so far (order-independent), recomputed
            # fresh each boundary — matches the validated offline analyzer, not a live running mean.
            bmean = [self._bsum[i] / self._bcnt[i] if self._bcnt[i] else 0.0 for i in range(_N_BUCKETS)]

            # per-key, per-epoch mean residual
            by_key_epoch: dict[str, dict[int, list[float]]] = {}
            for key, ep, b, loss in self._records:
                by_key_epoch.setdefault(key, {}).setdefault(ep, []).append(loss - bmean[b])

            per_key_epoch_mean: dict[str, dict[int, float]] = {
                key: {ep: sum(vals) / len(vals) for ep, vals in eps.items()}
                for key, eps in by_key_epoch.items()
            }
            # Merge in residuals restored from a prior run's checkpoint (already computed against
            # THAT run's bucket means, so they must be merged post-hoc, never re-derived here).
            # Only fills gaps — this run's own live records for a (key, epoch) always win.
            for key, eps in self._restored_residuals.items():
                dest = per_key_epoch_mean.setdefault(key, {})
                for ep, residual in eps.items():
                    dest.setdefault(ep, residual)

            all_residuals = [r for eps in per_key_epoch_mean.values() for r in eps.values()]
            all_residuals.sort()
            n = len(all_residuals)
            hi_thresh = all_residuals[min(int(n * self.hi_q), n - 1)] if n else 0.0
            lo_thresh = all_residuals[min(int(n * self.lo_q), n - 1)] if n else 0.0
            med = all_residuals[n // 2] if n else 0.0
            iqr = (all_residuals[min(int(n * 0.75), n - 1)] - all_residuals[min(int(n * 0.25), n - 1)]) if n else 0.0

            for key, eps in per_key_epoch_mean.items():
                epochs_sorted = sorted(eps)
                cur_residual = eps[epochs_sorted[-1]]

                # ---- improving test: recent half vs older half of the trend window ----
                window_epochs = [e for e in epochs_sorted if e > epoch - self.trend_window]
                improving = False
                if len(window_epochs) >= 4:
                    half = len(window_epochs) // 2
                    older = [eps[e] for e in window_epochs[:half]]
                    recent = [eps[e] for e in window_epochs[half:]]
                    older_mean = sum(older) / len(older)
                    recent_mean = sum(recent) / len(recent)
                    drop = older_mean - recent_mean
                    scatter = (sum((v - recent_mean) ** 2 for v in recent) / len(recent)) ** 0.5
                    noise_bar = max(self.improve_floor, scatter / max(len(recent) ** 0.5, 1))
                    improving = drop > max(self.improve_frac * abs(older_mean), noise_bar)

                is_high = cur_residual >= hi_thresh
                is_low = cur_residual <= lo_thresh
                is_wild_outlier = n > 0 and (cur_residual >= med + 3 * iqr)
                is_robust_outlier = n > 0 and (cur_residual >= med + 1.5 * iqr)

                # health bookkeeping: epochs comfortably out of the hard zone
                if not is_high:
                    self._healthy_epochs[key] = self._healthy_epochs.get(key, 0) + 1

                # exhausted: proved a good early run, then plateaued while still hard
                baseline = eps[epochs_sorted[0]]
                proved_good_run = (baseline - cur_residual) >= self.exhaust_drop_frac * max(abs(baseline), 1e-6)
                if proved_good_run and self._healthy_epochs.get(key, 0) >= self.healthy_min:
                    self._retired.add(key)

                stuck_vote = is_high and not improving and key not in self._retired
                clear_vote = not stuck_vote
                suspect_vote = (is_wild_outlier or (is_robust_outlier and len(epochs_sorted) >= 2)) \
                    and key not in self._confirmed_stuck and key not in self._retired
                exhaust_vote = key in self._retired and is_high

                self._stuck_votes[key] = self._stuck_votes.get(key, 0) + 1 if stuck_vote else 0
                self._clear_votes[key] = self._clear_votes.get(key, 0) + 1 if clear_vote else 0
                self._suspect_votes[key] = self._suspect_votes.get(key, 0) + 1 if suspect_vote else 0
                self._exhaust_votes[key] = self._exhaust_votes.get(key, 0) + 1 if exhaust_vote else 0

                if self._stuck_votes[key] >= self.persist_on:
                    self._confirmed_stuck.add(key)
                    self._stuck_epochs[key] = self._stuck_epochs.get(key, 0) + 1
                elif self._clear_votes[key] >= self.persist_off:
                    self._confirmed_stuck.discard(key)
                    self._stuck_epochs[key] = 0

                # ---- resolve multiplier + verdict ----
                if key in self._confirmed_stuck:
                    escalations = max(0, (self._stuck_epochs.get(key, 1) - 1) // self.escalate_every)
                    mult = max(self.stuck_floor, self.throttle_mult * (0.5 ** escalations))
                    verdict = "stuck"
                elif self._suspect_votes.get(key, 0) >= 1 and improving is False:
                    mult = self.suspect_mult
                    verdict = "suspect"
                elif self._exhaust_votes.get(key, 0) >= self.exhaust_on:
                    mult = self.exhausted_mult
                    verdict = "exhausted"
                elif (
                    epoch >= self.easy_from_epoch
                    and self._healthy_epochs.get(key, 0) >= self.healthy_min
                    and not is_high
                ):
                    mult = self.easy_mult
                    verdict = "healthy"
                elif is_high and improving:
                    mult = 1.0
                    verdict = "learning"
                else:
                    mult = 1.0
                    verdict = "normal"

                self._mult[key] = mult
                self.verdicts[key] = verdict

            counts: dict[str, int] = {}
            for v in self.verdicts.values():
                counts[v] = counts.get(v, 0) + 1
            summary = ", ".join(f"{v}={c}" for v, c in sorted(counts.items()))
            self._log(f"[adaptive-lr] window {epoch}: {len(self.verdicts)} image(s) tracked — {summary}")

            added = self._confirmed_stuck - self._last_reported_stuck
            removed = self._last_reported_stuck - self._confirmed_stuck
            if added:
                names = ", ".join(os.path.basename(k) for k in sorted(added))
                self._log(f"[adaptive-lr] window {epoch}: image(s) confirmed STUCK "
                          f"(persistently hard, not improving — check for bad/mislabeled data): "
                          f"{names} — throttling LR x{self.throttle_mult}")
            if removed:
                names = ", ".join(os.path.basename(k) for k in sorted(removed))
                self._log(f"[adaptive-lr] window {epoch}: no longer stuck: {names}")
            self._last_reported_stuck = set(self._confirmed_stuck)

            return self.verdicts
        except Exception as e:
            self._log(f"[adaptive-lr] epoch_boundary failed ({e})")
            return self.verdicts

    # ---- persistence -----------------------------------------------------------

    def state_dict(self) -> dict:
        """Compact snapshot of everything epoch_boundary needs to pick up where it left off.
        Deliberately does NOT include self._records (the raw per-step history) — those are only
        needed to RE-DERIVE per-key-epoch residual means, which are computed and stored here
        directly, so replaying a full per-step log isn't necessary the way it would be with no other
        per-step log to reconstruct from). This keeps the sidecar file small regardless of
        dataset size or run length."""
        flattened = {
            f"{k}\x1f{ep}": v
            for k, ep_map in self._all_per_key_epoch_residual().items()
            for ep, v in ep_map.items()
        }
        return {
            "version": 1,
            "mult": dict(self._mult),
            "verdicts": dict(self.verdicts),
            "stuck_votes": dict(self._stuck_votes),
            "clear_votes": dict(self._clear_votes),
            "suspect_votes": dict(self._suspect_votes),
            "exhaust_votes": dict(self._exhaust_votes),
            "stuck_epochs": dict(self._stuck_epochs),
            "confirmed_stuck": sorted(self._confirmed_stuck),
            "last_reported_stuck": sorted(self._last_reported_stuck),
            "healthy_epochs": dict(self._healthy_epochs),
            "retired": sorted(self._retired),
            "bucket_sum": list(self._bsum),
            "bucket_cnt": list(self._bcnt),
            # per-key-epoch residual means, flattened as "key\x1fepoch" -> mean, so a resumed run
            # can keep computing the trend-window improve test across the resume boundary.
            "per_key_epoch_residual": flattened,
        }

    def _all_per_key_epoch_residual(self) -> dict:
        """Every (key, epoch) mean residual known to this watcher — both from live records this
        run and any already-restored from a previous checkpoint. Used only when SAVING."""
        bmean = [self._bsum[i] / self._bcnt[i] if self._bcnt[i] else 0.0 for i in range(_N_BUCKETS)]
        out: dict[str, dict[int, list[float]]] = {}
        for key, ep, b, loss in self._records:
            out.setdefault(key, {}).setdefault(ep, []).append(loss - bmean[b])
        merged = {k: {ep: sum(v) / len(v) for ep, v in eps.items()} for k, eps in out.items()}
        for key, eps in self._restored_residuals.items():
            dest = merged.setdefault(key, {})
            for ep, residual in eps.items():
                dest.setdefault(ep, residual)
        return merged

    def load_state_dict(self, state: dict) -> None:
        """Restore watcher state saved by state_dict(). Never raises — a corrupt or missing
        sidecar just means the watch re-warms from scratch, same as a fresh run."""
        try:
            self._mult = dict(state.get("mult", {}))
            self.verdicts = dict(state.get("verdicts", {}))
            self._stuck_votes = dict(state.get("stuck_votes", {}))
            self._clear_votes = dict(state.get("clear_votes", {}))
            self._suspect_votes = dict(state.get("suspect_votes", {}))
            self._exhaust_votes = dict(state.get("exhaust_votes", {}))
            self._stuck_epochs = dict(state.get("stuck_epochs", {}))
            self._confirmed_stuck = set(state.get("confirmed_stuck", []))
            self._last_reported_stuck = set(state.get("last_reported_stuck", []))
            self._healthy_epochs = dict(state.get("healthy_epochs", {}))
            self._retired = set(state.get("retired", []))
            self._bsum = list(state.get("bucket_sum", self._bsum))
            self._bcnt = list(state.get("bucket_cnt", self._bcnt))
            # Un-flatten into (key -> epoch -> residual). These are merged into per_key_epoch_mean
            # at each epoch_boundary() call, never re-derived through a bucket lookup (the bucket
            # means that produced them belong to the PRIOR run and aren't reconstructable here).
            restored: dict[str, dict[int, float]] = {}
            for flat_key, residual in state.get("per_key_epoch_residual", {}).items():
                key, ep = flat_key.rsplit("\x1f", 1)
                restored.setdefault(key, {})[int(ep)] = float(residual)
            self._restored_residuals = restored
            self._log(f"[adaptive-lr] restored watch state: {len(self._mult)} tracked image(s), "
                      f"{len(self._confirmed_stuck)} currently stuck")
        except Exception as e:
            self._log(f"[adaptive-lr] load_state_dict failed ({e}) — watch re-warms from scratch")
