#!/usr/bin/env python
"""
Regression test for the gradient-accumulation zero_grad bug.

    python testing/test_grad_accum.py

ai-toolkit has TWO mutually exclusive accumulation knobs (toolkit/config_modules.py:463):

  gradient_accumulation: N        -> N micro-batches as a batch_list inside ONE
                                     hook_train_loop call
  gradient_accumulation_steps: N  -> ONE micro-batch per call, N calls per optimizer
                                     step (-1 = accumulate a whole epoch)

`SDTrainer.hook_train_loop` used to call `self.optimizer.zero_grad()` unconditionally at
the top. For the batch_list knob that is harmless (all micro-batches run inside the one
call), but for the repeated-call knob it wiped the previous micro-step's gradients on
every call, so the deferred optimizer step saw only the LAST micro-batch's gradient
scaled by 1/N instead of the mean over N.

PART 1 replicates the accumulation state machine -- the gating block from
BaseSDTrainProcess.hook_train_loop's driver and the zero_grad rule from
SDTrainer.hook_train_loop -- against a real torch optimizer, and checks what the
optimizer step actually sees. It is a simulation of the call sequence, not an
end-to-end training run.

PART 2 greps the two source files, so reverting the fix fails this test even though
PART 1 carries its own copy of the logic.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def simulate(gradient_accumulation=1, gradient_accumulation_steps=1, epoch_len=None,
             n_train_steps=12, guarded=True):
    """Drive the accumulation state machine and return the gradient seen by each
    optimizer.step(), plus the micro-batch gradients that fed it.

    guarded=False reproduces the pre-fix unconditional zero_grad().
    """
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.0)  # lr 0: we only inspect .grad

    # --- state, mirroring BaseSDTrainProcess.__init__ ---
    grad_accumulation_step = 1          # starts at 1 upstream
    is_grad_accumulation_step = False
    grad_accum_in_progress = False

    stepped = []        # gradient .grad held at each optimizer.step()
    fed = []            # micro-batch gradients accumulated since the last step
    micro = 0           # micro-batch counter -> gradient value (1.0, 2.0, 3.0, ...)
    pending = []

    for _ in range(n_train_steps):
        # ---- driver: default-on, before the batch fetch (BaseSDTrainProcess ~:2581) ----
        is_grad_accumulation_step = True

        # ---- driver: build the batch list (BaseSDTrainProcess ~:2601) ----
        batch_list = []
        for _ in range(gradient_accumulation):
            micro += 1
            batch_list.append(float(micro))
            if epoch_len is not None and micro % epoch_len == 0:
                # StopIteration -> epoch rollover (BaseSDTrainProcess ~:2628)
                if gradient_accumulation_steps == -1:
                    is_grad_accumulation_step = False
                    grad_accumulation_step = 0

        # ---- driver: decide whether this call steps (BaseSDTrainProcess ~:2640) ----
        if gradient_accumulation_steps == -1:
            pass                              # epoch handles it
        else:
            is_optimizer_step = grad_accumulation_step >= gradient_accumulation_steps
            is_grad_accumulation_step = not is_optimizer_step
            if is_optimizer_step:
                grad_accumulation_step = 0

        # ---- SDTrainer.hook_train_loop ----
        if guarded:
            if not grad_accum_in_progress:
                opt.zero_grad()
                pending = []
            grad_accum_in_progress = is_grad_accumulation_step
        else:
            opt.zero_grad()                   # the bug: unconditional
            pending = []

        n_accum = len(batch_list)
        if gradient_accumulation_steps > 1:
            n_accum *= gradient_accumulation_steps
        accum_scale = 1.0 / n_accum

        for g in batch_list:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad += g * accum_scale     # accelerator.backward(loss * accum_scale)
            pending.append(g)

        if not is_grad_accumulation_step:
            stepped.append(float(param.grad.item()))
            fed.append(list(pending))
            opt.step()
            opt.zero_grad(set_to_none=True)
            pending = []
            grad_accum_in_progress = False

        grad_accumulation_step += 1           # BaseSDTrainProcess ~:2852

    return stepped, fed


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_repeated_call_knob_averages():
    """gradient_accumulation_steps: 4 -> the step sees the MEAN of its 4 micro-batches."""
    stepped, fed = simulate(gradient_accumulation_steps=4, n_train_steps=12)
    assert len(stepped) == 3, f"expected 3 steps in 12 calls, got {len(stepped)}"
    for got, micro in zip(stepped, fed):
        assert len(micro) == 4, f"expected 4 micro-batches per step, got {micro}"
        want = sum(micro) / len(micro)
        assert approx(got, want), f"grad {got} != mean{micro} = {want}"
    print(f"  repeated-call knob averages ok  steps={stepped} from {fed}")


def test_repeated_call_knob_was_broken_before():
    """The same config pre-fix: only the last micro-batch survives, scaled by 1/N."""
    stepped, fed = simulate(gradient_accumulation_steps=4, n_train_steps=12, guarded=False)
    for got, micro in zip(stepped, fed):
        want_mean = sum(micro) / len(micro)
        assert not approx(got, want_mean), "pre-fix path unexpectedly produced the mean"
        assert approx(got, micro[-1] / 4.0), f"pre-fix grad {got} != last/4 = {micro[-1] / 4.0}"
    print(f"  pre-fix behaviour reproduced ok  steps={stepped} (= last micro-batch / 4)")


def test_batch_list_knob_unchanged():
    """gradient_accumulation: 4 was always correct; the fix must not alter it."""
    guarded, fed_g = simulate(gradient_accumulation=4, n_train_steps=3)
    unguarded, _ = simulate(gradient_accumulation=4, n_train_steps=3, guarded=False)
    assert guarded == unguarded, f"fix changed the batch-list knob: {guarded} vs {unguarded}"
    for got, micro in zip(guarded, fed_g):
        want = sum(micro) / len(micro)
        assert approx(got, want), f"grad {got} != mean{micro} = {want}"
    print(f"  batch-list knob unchanged ok  steps={guarded}")


def test_no_accumulation_unchanged():
    """The default (both knobs 1) steps every call, with or without the guard."""
    guarded, _ = simulate(n_train_steps=4)
    unguarded, _ = simulate(n_train_steps=4, guarded=False)
    assert guarded == unguarded == [1.0, 2.0, 3.0, 4.0], f"{guarded} / {unguarded}"
    print(f"  no-accumulation default unchanged ok  steps={guarded}")


def test_whole_epoch_knob_accumulates():
    """gradient_accumulation_steps: -1 accumulates a whole epoch, deliberately SUMMED
    (accum_scale stays 1.0 -- see the comment in SDTrainer.hook_train_loop)."""
    stepped, fed = simulate(gradient_accumulation_steps=-1, epoch_len=5, n_train_steps=10)
    assert len(stepped) == 2, f"expected 2 epoch steps, got {len(stepped)}: {stepped}"
    for got, micro in zip(stepped, fed):
        assert len(micro) == 5, f"expected 5 micro-batches per epoch, got {micro}"
        assert approx(got, sum(micro)), f"grad {got} != sum{micro} = {sum(micro)}"
    broken, _ = simulate(gradient_accumulation_steps=-1, epoch_len=5, n_train_steps=10,
                         guarded=False)
    assert broken != stepped, "the -1 path was supposed to be broken pre-fix too"
    print(f"  whole-epoch knob ok  steps={stepped} from {fed} (pre-fix: {broken})")


def test_sources_still_carry_the_fix():
    """Guard against a revert: PART 1 has its own copy of the logic, so it would keep
    passing even if the real source regressed."""
    p = os.path.join(REPO, 'extensions_built_in', 'sd_trainer', 'SDTrainer.py')
    src = open(p).read()
    assert 'if not self.grad_accum_in_progress:\n            self.optimizer.zero_grad()' in src, \
        f"{p}: guarded zero_grad is gone -- the accumulation bug is back"
    assert 'self.grad_accum_in_progress = self.is_grad_accumulation_step' in src, \
        f"{p}: the cycle flag is no longer updated in hook_train_loop"
    # the one unconditional zero_grad left must be the post-step one
    bare = [ln for ln in src.splitlines()
            if ln.strip() == 'self.optimizer.zero_grad()' and not ln.startswith('            ')]
    assert not bare, f"{p}: unguarded top-level zero_grad reintroduced: {bare}"

    p = os.path.join(REPO, 'jobs', 'process', 'BaseSDTrainProcess.py')
    src = open(p).read()
    assert 'self.grad_accum_in_progress = False' in src, \
        f"{p}: grad_accum_in_progress is not initialised/reset"
    assert src.count('self.grad_accum_in_progress = False') >= 3, \
        f"{p}: expected init + OOM reset + save reset, found {src.count('self.grad_accum_in_progress = False')}"
    print("  sources still carry the fix ok")


def main():
    print("gradient accumulation / zero_grad regression tests")
    test_no_accumulation_unchanged()
    test_batch_list_knob_unchanged()
    test_repeated_call_knob_averages()
    test_repeated_call_knob_was_broken_before()
    test_whole_epoch_knob_accumulates()
    test_sources_still_carry_the_fix()
    print("ALL OK")


if __name__ == '__main__':
    main()
