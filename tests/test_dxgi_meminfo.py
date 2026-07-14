import os
import unittest
from unittest import mock

from toolkit.memory_management import dxgi_meminfo, pin_manager

GIB = 1024 ** 3


class _FakeVM:
    def __init__(self, total):
        self.total = total


class _FakePsutil:
    def __init__(self, total):
        self._vm = _FakeVM(total)

    def virtual_memory(self):
        return self._vm


class DxgiHeadroomTests(unittest.TestCase):
    def setUp(self):
        self._saved_pinned = pin_manager.pinned_bytes_by_kind()
        pin_manager._LEDGER.clear()

        def _restore():
            pin_manager._LEDGER.clear()
            pin_manager._LEDGER.update(self._saved_pinned)

        self.addCleanup(_restore)

    def test_compute_non_local_headroom_normal_case(self):
        self.assertEqual(
            dxgi_meminfo.compute_non_local_headroom_bytes(16 * GIB, 10 * GIB, 2 * GIB),
            4 * GIB,
        )

    def test_compute_non_local_headroom_clamps_to_zero(self):
        self.assertEqual(
            dxgi_meminfo.compute_non_local_headroom_bytes(16 * GIB, 15 * GIB, 2 * GIB),
            0,
        )

    def test_compute_non_local_headroom_allows_zero_reserve(self):
        self.assertEqual(
            dxgi_meminfo.compute_non_local_headroom_bytes(16 * GIB, 10 * GIB, 0),
            6 * GIB,
        )

    def test_pinned_bytes_headroom_falls_back_to_legacy_proxy(self):
        with mock.patch.object(
            dxgi_meminfo, "query_non_local_video_memory_info", return_value=None
        ):
            with mock.patch.object(pin_manager, "_psutil", _FakePsutil(32 * GIB)):
                with mock.patch.dict(
                    os.environ,
                    {
                        "AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0.25",
                        "AI_TOOLKIT_WDDM_DXGI_DISABLE": "0",
                        "AI_TOOLKIT_WDDM_DXGI_CONTROL_DISABLE": "0",
                    },
                    clear=False,
                ):
                    pin_manager.register_pinned_bytes(3 * GIB)
                    self.assertEqual(pin_manager.pinned_bytes_headroom(0), 5 * GIB)

    def test_dxgi_headroom_does_not_subtract_pinned_ledger(self):
        reading = dxgi_meminfo.DxgiMemoryInfo(
            budget_bytes=16 * GIB,
            current_usage_bytes=10 * GIB,
            available_for_reservation_bytes=0,
            current_reservation_bytes=0,
        )
        with mock.patch.object(
            dxgi_meminfo, "query_non_local_video_memory_info", return_value=reading
        ):
            with mock.patch.dict(
                os.environ,
                {
                    "AI_TOOLKIT_WDDM_SPILL_RESERVE_GIB": "1.0",
                    "AI_TOOLKIT_WDDM_SPILL_RESERVE_PCT": "0",
                    "AI_TOOLKIT_WDDM_DXGI_DISABLE": "0",
                    "AI_TOOLKIT_WDDM_DXGI_CONTROL_DISABLE": "0",
                },
                clear=False,
            ):
                pin_manager.register_pinned_bytes(4 * GIB)
                # pct pinned to 0 -> reserve is exactly the 1 GiB floor, so the
                # focus is the double-subtract invariant (ledger not subtracted).
                self.assertEqual(pin_manager.pinned_bytes_headroom(0), 5 * GIB)

    def test_control_disable_keeps_legacy_proxy_for_control(self):
        reading = dxgi_meminfo.DxgiMemoryInfo(
            budget_bytes=64 * GIB,
            current_usage_bytes=1 * GIB,
            available_for_reservation_bytes=0,
            current_reservation_bytes=0,
        )
        with mock.patch.object(
            dxgi_meminfo, "query_non_local_video_memory_info", return_value=reading
        ):
            with mock.patch.object(pin_manager, "_psutil", _FakePsutil(32 * GIB)):
                with mock.patch.dict(
                    os.environ,
                    {
                        "AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0.25",
                        "AI_TOOLKIT_WDDM_DXGI_CONTROL_DISABLE": "1",
                    },
                    clear=False,
                ):
                    self.assertEqual(pin_manager.pinned_bytes_headroom(0), 8 * GIB)


class SpillReserveMarginTests(unittest.TestCase):
    def setUp(self):
        def _clear():
            pin_manager._SPILL_RESERVE_FLOOR_GIB_OVERRIDE = None
            pin_manager._SPILL_RESERVE_PCT_OVERRIDE = None
        _clear()
        self.addCleanup(_clear)

    def test_margin_is_pct_of_budget_when_pct_dominates(self):
        with mock.patch.dict(
            os.environ,
            {"AI_TOOLKIT_WDDM_SPILL_RESERVE_FLOOR_GIB": "2.0",
             "AI_TOOLKIT_WDDM_SPILL_RESERVE_PCT": "0.20"},
            clear=False,
        ):
            # 0.20 * 16 GiB = 3.2 GiB > 2 GiB floor.
            self.assertEqual(
                pin_manager.dxgi_spill_reserve_bytes(16 * GIB),
                int(0.20 * 16 * GIB),
            )

    def test_margin_is_floor_when_floor_dominates(self):
        with mock.patch.dict(
            os.environ,
            {"AI_TOOLKIT_WDDM_SPILL_RESERVE_FLOOR_GIB": "2.0",
             "AI_TOOLKIT_WDDM_SPILL_RESERVE_PCT": "0.20"},
            clear=False,
        ):
            # 0.20 * 8 GiB = 1.6 GiB < 2 GiB floor.
            self.assertEqual(pin_manager.dxgi_spill_reserve_bytes(8 * GIB), 2 * GIB)

    def test_margin_falls_back_to_floor_without_budget(self):
        with mock.patch.dict(
            os.environ,
            {"AI_TOOLKIT_WDDM_SPILL_RESERVE_FLOOR_GIB": "2.0",
             "AI_TOOLKIT_WDDM_SPILL_RESERVE_PCT": "0.20"},
            clear=False,
        ):
            self.assertEqual(pin_manager.dxgi_spill_reserve_bytes(None), 2 * GIB)
            self.assertEqual(pin_manager.dxgi_spill_reserve_bytes(0), 2 * GIB)

    def test_config_override_supersedes_env(self):
        with mock.patch.dict(
            os.environ,
            {"AI_TOOLKIT_WDDM_SPILL_RESERVE_FLOOR_GIB": "2.0",
             "AI_TOOLKIT_WDDM_SPILL_RESERVE_PCT": "0.20"},
            clear=False,
        ):
            pin_manager.set_spill_reserve_policy(floor_gib=3.0, pct=0.10)
            # config floor 3 GiB dominates 0.10 * 16 = 1.6 GiB.
            self.assertEqual(pin_manager.dxgi_spill_reserve_bytes(16 * GIB), 3 * GIB)


class SafeForControlTests(unittest.TestCase):
    def test_confident_auto_detect_is_safe(self):
        self.assertTrue(dxgi_meminfo.safe_for_control("single_nvidia"))
        self.assertTrue(dxgi_meminfo.safe_for_control("luid"))

    def test_manual_override_is_safe_but_flagged_manual(self):
        self.assertTrue(dxgi_meminfo.safe_for_control("env_override"))
        self.assertTrue(dxgi_meminfo.is_manual_control("env_override"))
        self.assertFalse(dxgi_meminfo.is_manual_control("single_nvidia"))

    def test_fallback_methods_are_not_safe(self):
        for method in ("sole_hardware_adapter", "global_conservative", "", None):
            self.assertFalse(dxgi_meminfo.safe_for_control(method))
            self.assertFalse(dxgi_meminfo.is_manual_control(method))


if __name__ == "__main__":
    unittest.main()
