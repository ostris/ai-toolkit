"""The governing device-free signal must reflect the whole card, not this process.

Background: torch.cuda.mem_get_info reports a per-process promise, not physical
availability. Measured on a 4070 with a second process holding 9 GiB, it claimed
4.70 GiB free while the card physically had 0.55 GiB. Planning residency against
that number silently crosses the WDDM dedicated cliff (no error, no allocator
retry -- just ~4x slower steps). These tests pin the reconciliation policy.
"""

import unittest

import torch

from toolkit.memory_management import nvml_meminfo, vram_budget

GIB = 1024 ** 3


class ReconcileFreeBytesTests(unittest.TestCase):
    """Pure policy: which free value governs."""

    def test_prefers_physical_when_driver_over_reports(self):
        # The production failure: driver promises 4.70 GiB, card has 0.55 GiB.
        governing = vram_budget.reconcile_free_bytes(
            int(4.70 * GIB), int(0.55 * GIB)
        )
        self.assertAlmostEqual(governing / GIB, 0.55, places=2)

    def test_falls_back_to_driver_when_nvml_unavailable(self):
        # No NVML (non-NVIDIA, no driver): must not crash, must not invent a
        # number -- prior behavior is preserved exactly.
        governing = vram_budget.reconcile_free_bytes(int(4.70 * GIB), None)
        self.assertAlmostEqual(governing / GIB, 4.70, places=2)

    def test_takes_the_pessimistic_signal_when_driver_is_lower(self):
        # The driver's promise is a real constraint too; never let NVML's larger
        # number talk us into exceeding it.
        governing = vram_budget.reconcile_free_bytes(int(1.0 * GIB), int(6.0 * GIB))
        self.assertAlmostEqual(governing / GIB, 1.0, places=2)

    def test_agrees_with_driver_when_uncontended(self):
        # Single-process case: the two signals match, so this policy costs no
        # residency versus the old behavior.
        governing = vram_budget.reconcile_free_bytes(int(10.6 * GIB), int(10.6 * GIB))
        self.assertAlmostEqual(governing / GIB, 10.6, places=2)

    def test_never_negative(self):
        self.assertEqual(vram_budget.reconcile_free_bytes(-5, None), 0)
        self.assertEqual(vram_budget.reconcile_free_bytes(GIB, -5), 0)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class NvmlSensorTests(unittest.TestCase):
    """The sensor itself, against the real driver."""

    def test_reports_plausible_physical_totals(self):
        info = nvml_meminfo.query_device_memory_info(0)
        if info is None:
            self.skipTest("NVML unavailable on this box")
        self.assertGreater(info.total_bytes, 0)
        self.assertEqual(info.used_bytes + info.free_bytes, info.total_bytes)
        card_total = int(torch.cuda.get_device_properties(0).total_memory)
        # Same card: NVML total is within a few percent of torch's (they differ
        # slightly on what they count as reserved by the driver).
        self.assertLess(abs(info.total_bytes - card_total) / card_total, 0.05)

    def test_physical_free_never_exceeds_the_card(self):
        free = nvml_meminfo.physical_free_bytes(0)
        if free is None:
            self.skipTest("NVML unavailable on this box")
        card_total = int(torch.cuda.get_device_properties(0).total_memory)
        self.assertGreaterEqual(free, 0)
        self.assertLessEqual(free, card_total)

    def test_governing_free_is_never_more_optimistic_than_the_driver(self):
        # The invariant that keeps us off the cliff: whatever the sensors say,
        # the governing number can only be <= what torch would have used.
        device = torch.device("cuda", 0)
        driver_free = int(torch.cuda.mem_get_info(device)[0])
        governing = vram_budget.device_free_bytes(device)
        self.assertLessEqual(governing, driver_free)
        self.assertGreaterEqual(governing, 0)

    def test_device_mem_info_is_a_drop_in_for_mem_get_info(self):
        device = torch.device("cuda", 0)
        free, total = vram_budget.device_mem_info(device)
        _driver_free, driver_total = torch.cuda.mem_get_info(device)
        self.assertEqual(total, int(driver_total))
        self.assertLessEqual(free, total)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class DeviceSnapshotTests(unittest.TestCase):
    def test_snapshot_free_is_the_governed_value(self):
        device = torch.device("cuda", 0)
        snap = vram_budget.DeviceSnapshot.capture(device)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.free, vram_budget.device_free_bytes(device))
        # non_torch is what the docstring claims: everything on the card that
        # isn't torch's allocator -- which now genuinely includes other processes.
        self.assertGreaterEqual(snap.non_torch, 0)


if __name__ == "__main__":
    unittest.main()
