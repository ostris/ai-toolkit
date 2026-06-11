"""Tests for scripts/remote/transport.py (U4).

Run directly: python testing/test_remote_transport.py

No network: every rsync/ssh invocation is either a pure command-builder
asserted on its argv, or executed through an injected fake runner returning
scripted CompletedProcess objects.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.remote import contract, transport
from scripts.remote.manifest import RunManifest
from scripts.remote.transport import Endpoint


def proc(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class DispatchRunner:
    """Fake runner: first handler whose needle appears in the joined argv wins."""

    def __init__(self, handlers, default=None):
        self.handlers = handlers      # list of (needle, CompletedProcess | callable)
        self.default = default if default is not None else proc("")
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        joined = " ".join(str(c) for c in cmd)
        for needle, resp in self.handlers:
            if needle in joined:
                return resp(cmd) if callable(resp) else resp
        return self.default

    def commands(self, needle):
        return [c for c in self.calls if needle in " ".join(str(x) for x in c)]


EP = Endpoint(host="1.2.3.4", port=10022)

NOW = 1768500000.0
RUN = "balfua_v3"
JOB_DIR = contract.remote_job_dir(RUN)  # /workspace/runs/balfua_v3/output/balfua_v3


def sample_line(step, idx, mtime, size=123456):
    return f"samples/1768412345678__{step:09d}_{idx}.jpg\t{size}\t{mtime:.4f}"

def ckpt_line(step, mtime, size=600000000):
    return f"{RUN}_{step:09d}.safetensors\t{size}\t{mtime:.4f}"

def opt_line(mtime, size=1200000000):
    return f"optimizer.pt\t{size}\t{mtime:.4f}"


def make_manifest(**kw):
    defaults = dict(run_name=RUN, job_name=RUN, prompt_count=12,
                    state=contract.RunState.RUNNING.value,
                    last_pulled_sample_step=0, last_pulled_checkpoint_step=0)
    defaults.update(kw)
    return RunManifest(**defaults)


def valid_safetensors_bytes():
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + b"\x00" * 16


def materializing_rsync(cmd):
    """Fake rsync for --files-from pulls: write valid safetensors container
    bytes for each pulled checkpoint so the post-pull integrity check passes."""
    dest = cmd[-1]
    list_paths = [a.split("=", 1)[1] for a in cmd if a.startswith("--files-from=")]
    if list_paths:
        with open(list_paths[0]) as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            if name.endswith(".safetensors"):
                path = os.path.join(dest, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(valid_safetensors_bytes())
    return proc("")


def pull_handlers(find_lines, pod_time=NOW, stat_lines=None, extra=None):
    """Standard scripted responses for a pull_artifacts cycle."""
    handlers = [
        ("date +%s", proc(f"{int(pod_time)}\n")),
        ("find ", proc("\n".join(find_lines) + ("\n" if find_lines else ""))),
        ("stat --printf", proc("\n".join(stat_lines or []) + ("\n" if stat_lines else ""))),
        ("cat ", proc("", returncode=1)),       # no sentinels present
        ("tail -c", proc("", returncode=1)),    # no log yet
        ("rsync", materializing_rsync),
    ]
    return (extra or []) + handlers


class TestCommandBuilders(unittest.TestCase):
    def test_ssh_base_args(self):
        self.assertEqual(
            transport.ssh_base_args(EP),
            ["ssh", "-p", "10022", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", f"UserKnownHostsFile={contract.SSH_KNOWN_HOSTS_FILE}",
             *contract.ssh_identity_args(),
             "root@1.2.3.4"],
        )

    def test_ssh_identity_args_when_key_exists(self):
        # BatchMode cannot prompt for a passphrase, so the dedicated
        # passphrase-less pipeline key must be offered explicitly when present
        # (live-validated failure mode: Permission denied (publickey)).
        with mock.patch("os.path.exists", return_value=True):
            args = contract.ssh_identity_args()
        self.assertEqual(args[:1], ["-i"])
        self.assertIn("IdentitiesOnly=yes", args)
        with mock.patch("os.path.exists", return_value=False):
            self.assertEqual(contract.ssh_identity_args(), [])

    def test_ssh_run_passes_text_capture_timeout(self):
        captured = {}
        def runner(cmd, **kw):
            captured["cmd"] = cmd
            captured.update(kw)
            return proc("ok")
        res = transport.ssh_run(EP, "echo hi", runner=runner)
        self.assertEqual(res.stdout, "ok")
        self.assertEqual(captured["cmd"][-1], "echo hi")
        self.assertTrue(captured["text"])
        self.assertTrue(captured["capture_output"])
        self.assertEqual(captured["timeout"], transport.SSH_TIMEOUT)

    def test_build_rsync_up_includes_filter_excludes(self):
        cmd = transport.build_rsync_up("/data/set/", EP, "/workspace/runs/r/dataset/")
        for pattern in ("._*", ".DS_Store", "_latent_cache"):
            self.assertIn(f"--exclude={pattern}", cmd)
        self.assertNotIn("--delete", cmd)
        self.assertEqual(cmd[-1], "root@1.2.3.4:/workspace/runs/r/dataset/")
        self.assertEqual(cmd[-2], "/data/set/")
        self.assertIn("-e", cmd)
        self.assertIn("ssh -p 10022", cmd[cmd.index("-e") + 1])

    def test_build_rsync_up_delete_within(self):
        cmd = transport.build_rsync_up("/repo/toolkit/", EP, "/app/ai-toolkit/toolkit/",
                                       delete_within=True)
        self.assertIn("--delete", cmd)

    def test_build_rsync_down_never_contains_delete(self):
        variants = [
            transport.build_rsync_down(EP, f"{JOB_DIR}/", "output/r/"),
            transport.build_rsync_down(EP, f"{JOB_DIR}/x.safetensors", "output/r/x.safetensors"),
            transport.build_rsync_down(EP, f"{JOB_DIR}/", "output/r/", excludes=("._*",)),
            transport.build_rsync_down(EP, f"{JOB_DIR}/", "output/r/", files_from="/tmp/list.txt"),
        ]
        for cmd in variants:
            self.assertFalse(any(a == "--delete" or a.startswith("--delete") for a in cmd),
                             f"--delete found in download command: {cmd}")
        self.assertIn("--files-from=/tmp/list.txt", variants[3])

    def test_rsync_execution_uses_long_timeout_ssh_keeps_short(self):
        captured = []

        def runner(cmd, **kw):
            captured.append((list(cmd), kw))
            return proc("")

        transport.upload_file(EP, __file__, "/workspace/runs/r/cfg.yaml",
                              runner=runner)
        rsync_kw = [kw for c, kw in captured if c[0] == "rsync"]
        self.assertTrue(rsync_kw)
        for kw in rsync_kw:
            self.assertEqual(kw.get("timeout"), transport.RSYNC_TIMEOUT)
        ssh_kw = [kw for c, kw in captured if c[0] == "ssh"]
        self.assertTrue(ssh_kw)
        for kw in ssh_kw:
            self.assertEqual(kw.get("timeout"), transport.SSH_TIMEOUT)


class TestUploadOverlay(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="aitk-overlay-test-")
        for entry in contract.OVERLAY_ALLOWLIST:
            path = os.path.join(self.repo, entry)
            if entry.endswith(".py"):
                with open(path, "w") as f:
                    f.write("# stub\n")
            else:
                os.makedirs(path)

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def _runner(self):
        def local_git(cmd):
            if "rev-parse" in cmd:
                return proc("localsha456\n")
            return proc("diff --git a/x b/x\n+local change\n")  # git diff
        return DispatchRunner([
            (f"git -C {contract.REMOTE_REPO_DIR} rev-parse HEAD", proc("imagecommit123\n")),
            ("git -C", local_git),
            ("rsync", proc("")),
        ])

    def test_overlay_delete_scoped_inside_allowlist_dirs_only(self):
        runner = self._runner()
        transport.upload_overlay(EP, self.repo, runner=runner)
        rsyncs = [c for c in runner.calls if c[0] == "rsync"]
        self.assertTrue(rsyncs, "no rsync commands issued")
        dir_entries = [e for e in contract.OVERLAY_ALLOWLIST
                       if os.path.isdir(os.path.join(self.repo, e))]
        deleted_targets = []
        for cmd in rsyncs:
            target = cmd[-1]
            # every overlay rsync lands inside the image repo, never elsewhere
            self.assertTrue(target.startswith(f"root@1.2.3.4:{contract.REMOTE_REPO_DIR}/"),
                            f"overlay rsync target escapes the repo: {target}")
            # no rsync may target the repo root itself
            self.assertNotEqual(target.rstrip("/"), f"root@1.2.3.4:{contract.REMOTE_REPO_DIR}",
                                f"root-level overlay rsync found: {cmd}")
            if "--delete" in cmd:
                deleted_targets.append(target)
                # --delete only inside an allowlisted subdir (trailing slash = scoped)
                self.assertTrue(
                    any(target == f"root@1.2.3.4:{contract.REMOTE_REPO_DIR}/{d}/" for d in dir_entries),
                    f"--delete used outside allowlisted dirs: {cmd}")
        self.assertEqual(len(deleted_targets), len(dir_entries),
                         "every allowlisted dir should be overlaid with scoped --delete")

    def test_overlay_returns_code_identity(self):
        runner = self._runner()
        identity = transport.upload_overlay(EP, self.repo, runner=runner)
        self.assertEqual(identity["image_repo_commit"], "imagecommit123")
        self.assertEqual(identity["overlay_git_sha"], "localsha456")
        expected = hashlib.sha256("diff --git a/x b/x\n+local change\n".encode()).hexdigest()
        self.assertEqual(identity["overlay_dirty_hash"], expected)
        # the image repo commit is read via ssh BEFORE any rsync runs
        first_ssh = runner.calls[0]
        self.assertEqual(first_ssh[0], "ssh")
        self.assertIn("rev-parse HEAD", " ".join(first_ssh))


class TestInventory(unittest.TestCase):
    def test_fixture_listing_parses_into_typed_entries(self):
        old = NOW - 3600
        lines = (
            [sample_line(250, i, old) for i in range(12)]
            + [sample_line(500, i, old) for i in range(12)]
            + [ckpt_line(250, old), opt_line(old)]
            + ["samples/._junk.jpg\t4096\t1768400000.0",   # AppleDouble: ignored
               "config.yaml\t900\t1768400000.0"]            # not an artifact: ignored
        )
        m = make_manifest()
        runner = DispatchRunner([("find ", proc("\n".join(lines) + "\n"))])
        inv = transport.list_remote_artifacts(EP, m, runner=runner)

        by_step = inv.samples_by_step()
        self.assertEqual(sorted(by_step), [250, 500])
        self.assertEqual(len(by_step[250]), 12)
        self.assertEqual(len(by_step[500]), 12)
        entry = by_step[250][0]
        self.assertEqual(entry.kind, "sample")
        self.assertEqual(entry.step, 250)
        self.assertEqual(entry.size, 123456)
        self.assertAlmostEqual(entry.mtime, old, places=2)

        ckpts = inv.checkpoints()
        self.assertEqual([c.step for c in ckpts], [250])
        self.assertEqual(ckpts[0].kind, "checkpoint")
        self.assertEqual(ckpts[0].size, 600000000)
        self.assertIsNotNone(inv.optimizer())
        self.assertIsNone(inv.final_checkpoint())
        # junk lines contributed nothing
        self.assertEqual(len(inv.entries), 26)

    def test_missing_job_dir_yields_empty_inventory(self):
        # absence is signalled by the explicit marker, NOT a nonzero exit
        runner = DispatchRunner(
            [("find ", proc(f"{transport.DIR_ABSENT_MARKER}\n"))])
        inv = transport.list_remote_artifacts(EP, make_manifest(), runner=runner)
        self.assertEqual(inv.entries, [])

    def test_listing_ssh_failure_raises_instead_of_empty(self):
        runner = DispatchRunner([("find ", proc("", returncode=255,
                                                stderr="Connection timed out"))])
        with self.assertRaises(RuntimeError) as ctx:
            transport.list_remote_artifacts(EP, make_manifest(), runner=runner)
        self.assertIn("artifact listing failed (ssh)", str(ctx.exception))
        self.assertIn("Connection timed out", str(ctx.exception))

    def test_find_command_wrapped_in_existence_check(self):
        cmd = transport._find_command(JOB_DIR)
        self.assertTrue(cmd.startswith(f"if [ -d '{JOB_DIR}' ]; then "))
        self.assertIn("find ", cmd)
        self.assertIn(f"else echo {transport.DIR_ABSENT_MARKER}; fi", cmd)


class TestPull(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-transport-test-")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _pull(self, m, handlers):
        runner = DispatchRunner(handlers)
        result = transport.pull_artifacts(EP, m, base_dir=self.base, now=NOW, runner=runner)
        return result, runner

    def test_complete_steps_pulled_and_watermarks_advance(self):
        old = NOW - 3600
        lines = ([sample_line(250, i, old) for i in range(12)]
                 + [sample_line(500, i, old) for i in range(12)]
                 + [ckpt_line(250, old), opt_line(old)])
        stat = [f"{JOB_DIR}/{RUN}_000000250.safetensors\t600000000"]
        m = make_manifest()
        result, runner = self._pull(m, pull_handlers(lines, stat_lines=stat))

        self.assertEqual(result.new_sample_steps, [250, 500])
        self.assertEqual(result.new_checkpoint_steps, [250])
        self.assertEqual(result.partial_steps, [])
        self.assertEqual(result.skipped_unstable, [])
        self.assertEqual(result.pulled_files, 26)  # 24 samples + 1 ckpt + optimizer
        self.assertEqual(m.last_pulled_sample_step, 500)
        self.assertEqual(m.last_pulled_checkpoint_step, 250)
        # download command shape: --files-from rsync rooted at the job dir,
        # landing in output/<run>/ (R23), never with --delete
        downloads = [c for c in runner.calls if c[0] == "rsync"]
        self.assertTrue(downloads)
        files_from = [c for c in downloads if any(a.startswith("--files-from=") for a in c)]
        self.assertEqual(len(files_from), 1)
        cmd = files_from[0]
        self.assertEqual(cmd[-2], f"root@1.2.3.4:{JOB_DIR}/")
        self.assertTrue(cmd[-1].endswith(os.path.join("output", RUN) + os.sep))
        for c in downloads:
            self.assertNotIn("--delete", c)
        # sentinels mirrored on EVERY pull (R25)
        self.assertTrue(runner.commands("exit_code"))
        self.assertTrue(runner.commands("tail -c"))

    def test_checkpoint_size_flap_skipped_unstable(self):
        old = NOW - 3600
        lines = [ckpt_line(250, old, size=600000000)]
        # second stat sees a DIFFERENT size: still being written
        stat = [f"{JOB_DIR}/{RUN}_000000250.safetensors\t612345678"]
        m = make_manifest()
        result, runner = self._pull(m, pull_handlers(lines, stat_lines=stat))
        self.assertEqual(result.new_checkpoint_steps, [])
        self.assertIn(f"{RUN}_000000250.safetensors", result.skipped_unstable)
        self.assertEqual(m.last_pulled_checkpoint_step, 0)  # watermark not advanced
        self.assertEqual(result.pulled_files, 0)
        self.assertIsNone(result.optimizer_pairing_step)

    def test_fresh_checkpoint_mtime_skipped(self):
        lines = [ckpt_line(250, NOW - 10)]  # 10s old < 60s threshold
        m = make_manifest()
        result, _ = self._pull(m, pull_handlers(lines))
        self.assertEqual(result.new_checkpoint_steps, [])
        self.assertIn(f"{RUN}_000000250.safetensors", result.skipped_unstable)

    def test_mtime_stability_uses_pod_clock_not_local(self):
        # Pod clock is 3600s AHEAD of the laptop. The checkpoint is 300s old
        # by the POD's clock (stable), but naively local_now - mtime would be
        # -3300 ("from the future" = fresh) and it would be wrongly skipped.
        pod_now = NOW + 3600
        lines = [ckpt_line(1000, pod_now - 300)]
        stat = [f"{JOB_DIR}/{RUN}_000001000.safetensors\t600000000"]
        m = make_manifest()
        result, _ = self._pull(m, pull_handlers(lines, pod_time=pod_now, stat_lines=stat))
        self.assertEqual(result.new_checkpoint_steps, [1000])
        self.assertEqual(result.skipped_unstable, [])

    def test_partial_batch_fresh_mtime_skipped(self):
        # 4/12 files, newest 30s old: mid-write batch, skip this cycle
        lines = [sample_line(750, i, NOW - 30) for i in range(4)]
        m = make_manifest()
        result, _ = self._pull(m, pull_handlers(lines))
        self.assertEqual(result.new_sample_steps, [])
        self.assertEqual(result.partial_steps, [])
        self.assertEqual(len(result.skipped_unstable), 4)
        self.assertEqual(m.last_pulled_sample_step, 0)

    def test_partial_batch_old_mtime_pulled_with_warning(self):
        # 4/12 files but the newest is 10 minutes old: no more files coming
        lines = [sample_line(750, i, NOW - 600) for i in range(4)]
        m = make_manifest()
        result, _ = self._pull(m, pull_handlers(lines))
        self.assertEqual(result.new_sample_steps, [])
        self.assertEqual(result.partial_steps, [750])
        self.assertEqual(result.pulled_files, 4)
        # partial steps never advance the watermark: re-pullable next cycle
        self.assertEqual(m.last_pulled_sample_step, 0)

    def test_partial_step_completing_later_reported_once(self):
        m = make_manifest()
        # cycle 1: 4/12 with old mtime → partial; watermark must NOT advance
        lines = [sample_line(750, i, NOW - 600) for i in range(4)]
        r1, _ = self._pull(m, pull_handlers(lines))
        self.assertEqual(r1.partial_steps, [750])
        self.assertEqual(m.last_pulled_sample_step, 0)
        # cycle 2: batch completed → reported once as complete, watermark moves
        lines = [sample_line(750, i, NOW - 600) for i in range(12)]
        r2, _ = self._pull(m, pull_handlers(lines))
        self.assertEqual(r2.new_sample_steps, [750])
        self.assertEqual(r2.partial_steps, [])
        self.assertEqual(m.last_pulled_sample_step, 750)
        # cycle 3: nothing new — no double-report after completion
        r3, _ = self._pull(m, pull_handlers(lines))
        self.assertEqual(r3.new_sample_steps, [])
        self.assertEqual(r3.partial_steps, [])

    def test_optimizer_paired_to_newest_pulled_checkpoint_only(self):
        old = NOW - 3600
        lines = [ckpt_line(1000, old), ckpt_line(1250, old + 100), opt_line(old + 100)]
        stat = [f"{JOB_DIR}/{RUN}_000001000.safetensors\t600000000",
                f"{JOB_DIR}/{RUN}_000001250.safetensors\t600000000"]
        m = make_manifest(last_pulled_checkpoint_step=750)
        result, runner = self._pull(m, pull_handlers(lines, stat_lines=stat))
        self.assertEqual(result.new_checkpoint_steps, [1000, 1250])
        self.assertEqual(result.optimizer_pairing_step, 1250)
        self.assertEqual(m.optimizer_pairing_step, 1250)
        # staged under the canonical 9-digit name, against 1250 only
        opt_rsyncs = [c for c in runner.calls
                      if c[0] == "rsync" and c[-1].endswith("optimizer_000001250.pt")]
        self.assertEqual(len(opt_rsyncs), 1)
        self.assertTrue(opt_rsyncs[0][-2].endswith("optimizer.pt"))
        self.assertFalse(any(c[-1].endswith("optimizer_000001000.pt")
                             for c in runner.calls if c[0] == "rsync"))

    def test_optimizer_not_paired_when_newest_checkpoint_was_skipped(self):
        old = NOW - 3600
        # 1250 is fresh/unstable; optimizer.pt likely belongs to it, not 1000
        lines = [ckpt_line(1000, old), ckpt_line(1250, NOW - 5), opt_line(NOW - 5)]
        stat = [f"{JOB_DIR}/{RUN}_000001000.safetensors\t600000000"]
        m = make_manifest()
        result, _ = self._pull(m, pull_handlers(lines, stat_lines=stat))
        self.assertEqual(result.new_checkpoint_steps, [1000])
        self.assertIsNone(result.optimizer_pairing_step)
        self.assertIsNone(m.optimizer_pairing_step)


class TestSafetensorsValidation(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-transport-test-")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _write(self, name, data):
        path = os.path.join(self.base, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_fabricated_valid_header_passes(self):
        path = self._write("ok.safetensors", valid_safetensors_bytes())
        self.assertTrue(transport.validate_safetensors_container(path))

    def test_truncated_file_rejected(self):
        path = self._write("short.safetensors", b"\x10\x00\x01")
        self.assertFalse(transport.validate_safetensors_container(path))

    def test_header_length_past_eof_rejected(self):
        # claims a 4096-byte header but the file ends after 4 bytes of it
        data = (4096).to_bytes(8, "little") + b"{\"a\""
        path = self._write("lying.safetensors", data)
        self.assertFalse(transport.validate_safetensors_container(path))

    def test_non_json_header_rejected(self):
        garbage = b"not json at all!"
        data = len(garbage).to_bytes(8, "little") + garbage
        path = self._write("garbage.safetensors", data)
        self.assertFalse(transport.validate_safetensors_container(path))

    def test_pull_rejects_corrupt_checkpoint_and_keeps_valid_one(self):
        old = NOW - 3600
        lines = [ckpt_line(1000, old), ckpt_line(1250, old)]
        stat = [f"{JOB_DIR}/{RUN}_000001000.safetensors\t600000000",
                f"{JOB_DIR}/{RUN}_000001250.safetensors\t600000000"]
        out_dir = os.path.join(self.base, "output", RUN)

        def fake_rsync(cmd):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{RUN}_000001000.safetensors"), "wb") as f:
                f.write(valid_safetensors_bytes())
            with open(os.path.join(out_dir, f"{RUN}_000001250.safetensors"), "wb") as f:
                f.write(b"\x00\x01\x02")  # truncated garbage
            return proc("")

        m = make_manifest()
        runner = DispatchRunner(pull_handlers(
            lines, stat_lines=stat, extra=[("--files-from", fake_rsync)]))
        result = transport.pull_artifacts(EP, m, base_dir=self.base, now=NOW,
                                          runner=runner)
        # only the valid checkpoint counts; the corrupt one is deleted and
        # left re-pullable (watermark stops at 1000, not 1250)
        self.assertEqual(result.new_checkpoint_steps, [1000])
        self.assertEqual(m.last_pulled_checkpoint_step, 1000)
        self.assertEqual(result.pulled_files, 1)
        self.assertTrue(any("failed integrity check" in s
                            for s in result.skipped_unstable))
        self.assertTrue(os.path.exists(
            os.path.join(out_dir, f"{RUN}_000001000.safetensors")))
        self.assertFalse(os.path.exists(
            os.path.join(out_dir, f"{RUN}_000001250.safetensors")))


class TestSentinelMirror(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-transport-test-")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_exit_code_mirrored_and_recorded(self):
        runner = DispatchRunner([
            ("cat '/workspace/runs/balfua_v3/exit_code'", proc("0\n")),
            ("cat '/workspace/runs/balfua_v3/timed_out'", proc("", returncode=1)),
            ("tail -c", proc("step 1000 loss 0.12\n")),
            ("rsync", proc("")),  # loss_log.db mirror (terminal because sentinel found)
        ])
        m = make_manifest()
        info = transport.mirror_sentinels(EP, m, base_dir=self.base, runner=runner)
        self.assertEqual(info["exit_code"], "0")
        self.assertFalse(info["timed_out"])
        self.assertEqual(m.last_sentinel, "0")
        local = os.path.join(self.base, "runs", RUN, "exit_code")
        with open(local) as f:
            self.assertEqual(f.read(), "0\n")
        with open(os.path.join(self.base, "runs", RUN, transport.LOG_TAIL_FILE)) as f:
            self.assertIn("step 1000", f.read())
        # terminal pull → loss_log.db mirrored, and never with --delete
        loss_rsyncs = [c for c in runner.calls
                       if c[0] == "rsync" and "loss_log.db" in " ".join(c)]
        self.assertEqual(len(loss_rsyncs), 1)
        self.assertNotIn("--delete", loss_rsyncs[0])

    def test_timed_out_wins_over_exit_code(self):
        runner = DispatchRunner([
            ("exit_code", proc("137\n")),
            ("timed_out", proc("1768500000\n")),
            ("tail -c", proc("", returncode=1)),
            ("rsync", proc("")),
        ])
        m = make_manifest()
        info = transport.mirror_sentinels(EP, m, base_dir=self.base, runner=runner)
        self.assertTrue(info["timed_out"])
        self.assertEqual(m.last_sentinel, "timed_out")
        self.assertTrue(os.path.exists(os.path.join(self.base, "runs", RUN, "timed_out")))

    def test_absent_sentinels_tolerated(self):
        runner = DispatchRunner([
            ("cat ", proc("", returncode=1)),
            ("tail -c", proc("", returncode=1)),
        ])
        m = make_manifest()  # state RUNNING → not terminal, no loss db attempt
        info = transport.mirror_sentinels(EP, m, base_dir=self.base, runner=runner)
        self.assertIsNone(info["exit_code"])
        self.assertFalse(info["timed_out"])
        self.assertIsNone(m.last_sentinel)
        self.assertFalse(os.path.exists(os.path.join(self.base, "runs", RUN, "exit_code")))
        self.assertEqual(runner.commands("rsync"), [])


class TestConfigDrift(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-transport-test-")
        self.run_dir = os.path.join(self.base, "runs", RUN)
        os.makedirs(self.run_dir)
        self.config_path = os.path.join(self.run_dir, transport.DERIVED_CONFIG_FILE)
        with open(self.config_path, "w") as f:
            f.write("job: extension\nconfig:\n  name: balfua_v3\n")
        with open(self.config_path, "rb") as f:
            self.good_hash = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_no_drift_when_hash_matches(self):
        m = make_manifest(config_hash=self.good_hash)
        self.assertFalse(transport.check_config_drift(m, base_dir=self.base))

    def test_tampered_config_drifts(self):
        m = make_manifest(config_hash=self.good_hash)
        with open(self.config_path, "a") as f:
            f.write("  # tampered after launch\n")
        self.assertTrue(transport.check_config_drift(m, base_dir=self.base))

    def test_missing_derived_config_counts_as_drift(self):
        m = make_manifest(config_hash=self.good_hash)
        os.unlink(self.config_path)
        self.assertTrue(transport.check_config_drift(m, base_dir=self.base))

    def test_no_recorded_hash_is_not_drift(self):
        m = make_manifest(config_hash=None)
        self.assertFalse(transport.check_config_drift(m, base_dir=self.base))


class TestPodClock(unittest.TestCase):
    def test_offset_is_pod_minus_local(self):
        runner = DispatchRunner([("date +%s", proc(f"{int(NOW) + 3600}\n"))])
        offset = transport.pod_clock_offset(EP, runner=runner, now=NOW)
        self.assertAlmostEqual(offset, 3600.0, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
