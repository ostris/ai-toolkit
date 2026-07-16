import os
import re
import unittest
from pathlib import Path
from unittest import mock

from toolkit.memory_management import pin_manager


GIB = 1024 ** 3
REPO_ROOT = Path(__file__).resolve().parents[1]


class PinManagerTests(unittest.TestCase):
    def setUp(self):
        pin_manager.reset_for_tests()

    def tearDown(self):
        pin_manager.reset_for_tests()

    def test_ledger_tracks_kinds(self):
        pin_manager.register_pinned_bytes(100, "weights")
        pin_manager.register_pinned_bytes(50, "bounce")
        pin_manager.release_pinned_bytes(25, "weights")

        self.assertEqual(pin_manager.total_pinned_bytes(), 125)
        self.assertEqual(
            pin_manager.pinned_bytes_by_kind(),
            {"weights": 75, "bounce": 50},
        )

    def test_reserve_reduces_available_grant(self):
        with mock.patch.dict(
            os.environ,
            {"AI_TOOLKIT_PIN_HOST_CACHE_RESERVE_GIB": "2.0"},
        ):
            with mock.patch.object(
                pin_manager, "pinned_bytes_headroom", return_value=8 * GIB
            ):
                self.assertEqual(
                    pin_manager.available_for_pin(mode="training"), 6 * GIB
                )


    def test_release_clamps_within_kind_only(self):
        # An unmatched release (consumer releasing bytes it never registered,
        # e.g. pinning was disabled) must not drain other consumers' entries.
        pin_manager.register_pinned_bytes(100, "weights")
        pin_manager.release_pinned_bytes(50, "bounce")
        self.assertEqual(pin_manager.pinned_bytes_by_kind(), {"weights": 100})

    def test_release_handle_is_idempotent(self):
        pin_manager.register_pinned_bytes(64, "save_stager")
        handle = pin_manager.PinHandle(
            tensor=None, nbytes=64, kind="save_stager", pinned=True
        )
        pin_manager.release(handle)
        self.assertEqual(pin_manager.total_pinned_bytes(), 0)
        pin_manager.release(handle)  # second release must be a no-op
        self.assertEqual(pin_manager.total_pinned_bytes(), 0)

    def test_failed_registered_release_retains_handle_and_ledger_for_retry(self):
        tensor = mock.Mock()
        handle = pin_manager.PinHandle(
            tensor=tensor,
            nbytes=64,
            kind="weights",
            pinned=True,
            mechanism="register",
        )
        pin_manager.register_pinned_bytes(64, "weights")

        with mock.patch.object(
            pin_manager, "unpin_tensor_in_place", return_value=False
        ):
            with self.assertRaises(pin_manager.PinReleaseError):
                pin_manager.release(handle)

        self.assertTrue(handle.pinned)
        self.assertEqual(handle.nbytes, 64)
        self.assertIs(handle.tensor, tensor)
        self.assertEqual(pin_manager.total_pinned_bytes(), 64)

        def succeed(_tensor, kind):
            pin_manager.release_pinned_bytes(64, kind)
            return True

        with mock.patch.object(
            pin_manager, "unpin_tensor_in_place", side_effect=succeed
        ):
            pin_manager.release(handle)

        self.assertFalse(handle.pinned)
        self.assertEqual(handle.nbytes, 0)
        self.assertEqual(pin_manager.total_pinned_bytes(), 0)


class PinConformanceTests(unittest.TestCase):
    """No direct pinning outside the pin manager.

    Every page-lock in the memory subsystem must route through pin_manager so
    the ledger stays authoritative. Scope is the offload subsystem + async
    save; upstream dataloader pin_memory usage is out of scope."""

    SCOPED_FILES = (
        "toolkit/async_save.py",
        "toolkit/memory_management/canonical_arena.py",
        "toolkit/memory_management/manager.py",
        "toolkit/memory_management/manager_modules.py",
        "toolkit/memory_management/arena_offload/construction.py",
        "toolkit/memory_management/arena_offload/transfer.py",
    )
    PATTERN = re.compile(r"pin_memory\s*=\s*True|\.pin_memory\(\)")

    def test_no_direct_pinning_outside_pin_manager(self):
        offenders = []
        for rel in self.SCOPED_FILES:
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if self.PATTERN.search(line) and "noqa: pin-manager" not in line:
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            "direct pinning outside pin_manager (route through pin_alloc / "
            "pin_tensor_in_place):\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
