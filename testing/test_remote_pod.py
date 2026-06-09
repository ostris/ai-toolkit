"""Tests for scripts/remote/pod.py (U3) — RunPod control plane wrapper.

Run directly: python testing/test_remote_pod.py

The runpod SDK is NEVER imported here: every pod.py function takes sdk=None
and the tests inject a unittest.mock.MagicMock module. No network.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.remote import contract, pod


FAKE_PUBKEY = "ssh-ed25519 AAAATESTKEY derrick@mac"

# A ready pod the way runpod's get_pod returns it: desiredStatus RUNNING +
# runtime.ports with the public-IP 22/tcp mapping.
RAW_READY_POD = {
    "id": "pod123",
    "desiredStatus": "RUNNING",
    "costPerHr": 1.89,
    "machine": {"gpuDisplayName": "NVIDIA A100 80GB PCIe"},
    "runtime": {
        "ports": [
            {"ip": "100.65.0.2", "isIpPublic": False, "privatePort": 8675,
             "publicPort": 60123, "type": "http"},
            {"ip": "194.26.196.10", "isIpPublic": True, "privatePort": 22,
             "publicPort": 10422, "type": "tcp"},
        ]
    },
}

RAW_NOT_READY_POD = {
    "id": "pod123",
    "desiredStatus": "RUNNING",
    "costPerHr": 1.89,
    "machine": {"gpuDisplayName": "NVIDIA A100 80GB PCIe"},
    "runtime": None,  # no ports exposed yet
}


class PodTestBase(unittest.TestCase):
    def setUp(self):
        env_patch = mock.patch.dict(os.environ, {
            "RUNPOD_STOP_API_KEY": "rps_scoped_key",
            "HF_TOKEN": "hf_test_token",
        })
        env_patch.start()
        self.addCleanup(env_patch.stop)
        key_patch = mock.patch.object(pod, "read_public_key",
                                      return_value=FAKE_PUBKEY)
        key_patch.start()
        self.addCleanup(key_patch.stop)


class TestCreatePod(PodTestBase):
    def test_provision_happy_path(self):
        sdk = mock.MagicMock()
        sdk.create_pod.return_value = {"id": "pod123", "desiredStatus": "CREATED"}
        sdk.get_pod.return_value = RAW_READY_POD

        info = pod.create_pod("balfua_v3", disk_gb=88, sdk=sdk)

        self.assertEqual(info.pod_id, "pod123")
        self.assertEqual(info.status, "RUNNING")
        self.assertEqual(info.ssh_host, "194.26.196.10")
        self.assertEqual(info.ssh_port, 10422)
        self.assertEqual(info.gpu_type, "NVIDIA A100 80GB PCIe")
        self.assertAlmostEqual(info.hourly_rate, 1.89)

        kwargs = sdk.create_pod.call_args.kwargs
        self.assertEqual(kwargs["name"], "aitk-balfua_v3")
        self.assertEqual(kwargs["gpu_type_id"], pod.DEFAULT_GPU)
        self.assertEqual(kwargs["cloud_type"], "SECURE")
        self.assertEqual(kwargs["ports"], "22/tcp")
        self.assertEqual(kwargs["volume_in_gb"], 88)
        self.assertEqual(kwargs["volume_mount_path"], "/workspace")
        # R27: the start-command override is on the create request
        self.assertEqual(kwargs["docker_args"], pod.container_start_command())
        # env carries the SSH key, HF token passthrough, and the stop key
        self.assertEqual(kwargs["env"]["PUBLIC_KEY"], FAKE_PUBKEY)
        self.assertEqual(kwargs["env"]["HF_TOKEN"], "hf_test_token")
        self.assertEqual(kwargs["env"]["RUNPOD_STOP_KEY"], "rps_scoped_key")

    def test_create_requests_public_ip_support(self):
        sdk = mock.MagicMock()
        sdk.create_pod.return_value = {"id": "pod123"}
        sdk.get_pod.return_value = RAW_READY_POD
        pod.create_pod("r1", disk_gb=80, sdk=sdk)
        self.assertIs(sdk.create_pod.call_args.kwargs["support_public_ip"], True)

    def test_invalid_run_name_rejected_before_any_call(self):
        sdk = mock.MagicMock()
        with self.assertRaises(ValueError):
            pod.create_pod("bad name; rm -rf", disk_gb=80, sdk=sdk)
        self.assertFalse(sdk.method_calls)

    def test_image_tag_param_beats_env_beats_default(self):
        sdk = mock.MagicMock()
        sdk.create_pod.return_value = {"id": "p"}
        sdk.get_pod.return_value = RAW_READY_POD
        with mock.patch.dict(os.environ, {"AITK_REMOTE_IMAGE": "ostris/aitoolkit:0.9.11"}):
            pod.create_pod("r1", disk_gb=80, sdk=sdk)
            self.assertEqual(sdk.create_pod.call_args.kwargs["image_name"],
                             "ostris/aitoolkit:0.9.11")
            pod.create_pod("r1", disk_gb=80, image_tag="ostris/aitoolkit:0.9.12", sdk=sdk)
            self.assertEqual(sdk.create_pod.call_args.kwargs["image_name"],
                             "ostris/aitoolkit:0.9.12")

    def test_out_of_stock_fails_fast_naming_gpu(self):
        sdk = mock.MagicMock()
        sdk.create_pod.side_effect = Exception(
            "There are no longer any instances available with the requested "
            "specifications. Please refresh and try again."
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(pod.OutOfStockError) as ctx:
                pod.create_pod("r1", disk_gb=80, sdk=sdk)
        self.assertIn(pod.DEFAULT_GPU, str(ctx.exception))
        self.assertEqual(sdk.create_pod.call_count, 1)  # no implicit retries
        sdk.terminate_pod.assert_not_called()

    def test_gpu_fallback_provisions_second_type(self):
        h100_id = pod.GPU_ALIASES["H100"]
        raw_h100 = {
            "id": "podH",
            "desiredStatus": "RUNNING",
            "costPerHr": 2.99,
            "machine": {"gpuDisplayName": "NVIDIA H100 80GB HBM3"},
            "runtime": {"ports": [
                {"ip": "194.26.196.99", "isIpPublic": True, "privatePort": 22,
                 "publicPort": 11022, "type": "tcp"},
            ]},
        }

        def create_side_effect(**kwargs):
            if kwargs["gpu_type_id"] == pod.DEFAULT_GPU:
                raise Exception("no longer any instances available")
            return {"id": "podH", "desiredStatus": "CREATED"}

        sdk = mock.MagicMock()
        sdk.create_pod.side_effect = create_side_effect
        sdk.get_pod.return_value = raw_h100

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            info = pod.create_pod("r1", disk_gb=80, gpu_fallback=("H100",), sdk=sdk)

        self.assertEqual(sdk.create_pod.call_count, 2)
        self.assertEqual(sdk.create_pod.call_args_list[1].kwargs["gpu_type_id"], h100_id)
        # PodInfo.gpu_type reflects the actually provisioned GPU
        self.assertEqual(info.gpu_type, "NVIDIA H100 80GB HBM3")
        self.assertEqual(info.pod_id, "podH")

    def test_all_fallbacks_exhausted_names_every_type(self):
        sdk = mock.MagicMock()
        sdk.create_pod.side_effect = Exception("no instances available")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(pod.OutOfStockError) as ctx:
                pod.create_pod("r1", disk_gb=80, gpu_fallback=("H100",), sdk=sdk)
        msg = str(ctx.exception)
        self.assertIn(pod.DEFAULT_GPU, msg)
        self.assertIn(pod.GPU_ALIASES["H100"], msg)

    def test_non_stock_error_raises_pod_error(self):
        sdk = mock.MagicMock()
        sdk.create_pod.side_effect = Exception("invalid api key")
        with self.assertRaises(pod.PodError):
            pod.create_pod("r1", disk_gb=80, sdk=sdk)


class TestDryRun(PodTestBase):
    def test_dry_run_prints_request_and_never_calls_sdk(self):
        sdk = mock.MagicMock()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            info = pod.create_pod("r1", disk_gb=88, dry_run=True, sdk=sdk)
        out = stdout.getvalue()

        self.assertEqual(info.pod_id, "DRY_RUN")
        self.assertEqual(info.status, "DRY_RUN")
        self.assertFalse(sdk.method_calls)  # NO sdk calls at all
        self.assertFalse(sdk.mock_calls)

        # The printed request is real: parse the JSON body back out
        body = out[out.index("{"):]
        request = json.loads(body[:body.rindex("}") + 1])
        self.assertEqual(request["ports"], "22/tcp")
        self.assertEqual(request["volume_in_gb"], 88)
        self.assertEqual(request["gpu_type_id"], pod.DEFAULT_GPU)
        self.assertIn("/start.sh", request["docker_args"])
        # secrets are redacted in the printed request
        self.assertNotIn("rps_scoped_key", out)
        self.assertNotIn("hf_test_token", out)


class TestGetPodInfo(PodTestBase):
    def test_parses_ssh_endpoint_from_runtime_ports(self):
        sdk = mock.MagicMock()
        sdk.get_pod.return_value = RAW_READY_POD
        info = pod.get_pod_info("pod123", sdk=sdk)
        self.assertEqual(info.ssh_host, "194.26.196.10")
        self.assertEqual(info.ssh_port, 10422)
        self.assertEqual(info.status, "RUNNING")
        self.assertAlmostEqual(info.hourly_rate, 1.89)

    def test_gone_when_api_returns_none(self):
        sdk = mock.MagicMock()
        sdk.get_pod.return_value = None
        info = pod.get_pod_info("podX", sdk=sdk)
        self.assertEqual(info.status, "GONE")
        self.assertEqual(info.pod_id, "podX")

    def test_gone_when_api_404s(self):
        sdk = mock.MagicMock()
        sdk.get_pod.side_effect = Exception("404 Client Error: pod not found")
        info = pod.get_pod_info("podX", sdk=sdk)
        self.assertEqual(info.status, "GONE")

    def test_other_errors_raise_pod_error(self):
        sdk = mock.MagicMock()
        sdk.get_pod.side_effect = Exception("rate limited")
        with self.assertRaises(pod.PodError):
            pod.get_pod_info("podX", sdk=sdk)

    def test_no_public_port_means_no_endpoint(self):
        sdk = mock.MagicMock()
        raw = dict(RAW_READY_POD)
        raw["runtime"] = {"ports": [
            {"ip": "10.0.0.5", "isIpPublic": False, "privatePort": 22,
             "publicPort": 22, "type": "tcp"},  # proxy-only: not usable
        ]}
        sdk.get_pod.return_value = raw
        info = pod.get_pod_info("pod123", sdk=sdk)
        self.assertIsNone(info.ssh_host)
        self.assertIsNone(info.ssh_port)


class TestWaitForReady(PodTestBase):
    def test_readiness_timeout_terminates_pod(self):
        sdk = mock.MagicMock()
        sdk.get_pod.return_value = RAW_NOT_READY_POD  # never exposes SSH
        with self.assertRaises(pod.SshTimeoutError) as ctx:
            pod.wait_for_ready("pod123", timeout_s=30, poll_s=10, sdk=sdk,
                               _sleep=lambda s: None)
        sdk.terminate_pod.assert_called_once_with("pod123")
        self.assertIn("pod123", str(ctx.exception))

    def test_probe_failure_terminates_pod(self):
        sdk = mock.MagicMock()
        sdk.get_pod.return_value = RAW_READY_POD  # SSH resolves...
        probe_calls = []

        def failing_probe(host, port):
            probe_calls.append((host, port))
            return False  # ...but the rsync probe never passes

        with self.assertRaises(pod.SshTimeoutError):
            pod.wait_for_ready("pod123", timeout_s=30, poll_s=10,
                               probe=failing_probe, sdk=sdk,
                               _sleep=lambda s: None)
        sdk.terminate_pod.assert_called_once_with("pod123")
        self.assertEqual(probe_calls[0], ("194.26.196.10", 10422))
        self.assertGreater(len(probe_calls), 1)  # it kept retrying

    def test_ready_returns_info_without_terminating(self):
        sdk = mock.MagicMock()
        sdk.get_pod.side_effect = [RAW_NOT_READY_POD, RAW_READY_POD]
        info = pod.wait_for_ready("pod123", timeout_s=900, poll_s=10,
                                  probe=lambda h, p: True, sdk=sdk,
                                  _sleep=lambda s: None)
        self.assertEqual(info.ssh_host, "194.26.196.10")
        self.assertEqual(info.ssh_port, 10422)
        sdk.terminate_pod.assert_not_called()

    def test_pod_gone_while_waiting_raises_without_terminate(self):
        sdk = mock.MagicMock()
        sdk.get_pod.return_value = None
        with self.assertRaises(pod.PodError) as ctx:
            pod.wait_for_ready("pod123", timeout_s=30, poll_s=10, sdk=sdk,
                               _sleep=lambda s: None)
        self.assertNotIsInstance(ctx.exception, pod.SshTimeoutError)
        sdk.terminate_pod.assert_not_called()

    def test_default_probe_runs_ssh_batchmode_rsync_noop(self):
        completed = mock.MagicMock()
        completed.returncode = 0
        with mock.patch.object(pod.subprocess, "run",
                               return_value=completed) as run:
            self.assertTrue(pod.default_ssh_probe("1.2.3.4", 10422))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("BatchMode=yes", cmd)
        self.assertIn("10422", cmd)
        self.assertIn("root@1.2.3.4", cmd)
        self.assertIn("rsync --version", cmd)


class TestStopTerminateRescue(PodTestBase):
    def test_stop_pod_calls_sdk(self):
        sdk = mock.MagicMock()
        pod.stop_pod("pod123", sdk=sdk)
        sdk.stop_pod.assert_called_once_with("pod123")

    def test_terminate_idempotent_on_404(self):
        sdk = mock.MagicMock()
        sdk.terminate_pod.side_effect = Exception("404: pod not found")
        pod.terminate_pod("gone-pod", sdk=sdk)  # must not raise

    def test_terminate_idempotent_on_does_not_exist(self):
        sdk = mock.MagicMock()
        sdk.terminate_pod.side_effect = Exception("pod does not exist")
        pod.terminate_pod("gone-pod", sdk=sdk)  # must not raise

    def test_terminate_passes_real_errors_through(self):
        sdk = mock.MagicMock()
        sdk.terminate_pod.side_effect = Exception("internal server error")
        with self.assertRaises(pod.PodError):
            pod.terminate_pod("pod123", sdk=sdk)

    def test_start_zero_gpu_resumes_with_zero_count(self):
        sdk = mock.MagicMock()
        sdk.get_pod.return_value = RAW_READY_POD
        info = pod.start_zero_gpu("pod123", sdk=sdk)
        sdk.resume_pod.assert_called_once_with("pod123", gpu_count=0)
        self.assertEqual(info.pod_id, "pod123")


class TestProviderStrings(PodTestBase):
    def test_self_stop_script_uses_scoped_key_only(self):
        script = pod.self_stop_script()
        self.assertIn("$RUNPOD_POD_ID", script)
        self.assertIn("$RUNPOD_STOP_KEY", script)
        # NEVER the full account key name on the pod (R21)
        self.assertNotIn("RUNPOD_API_KEY", script)
        # runpodctl is not in the image: self-stop must be curl-based
        self.assertIn("curl", script)
        self.assertNotIn("runpodctl", script)
        # contract owns the uploaded filename
        self.assertEqual(contract.SELF_STOP_SCRIPT, "self_stop.sh")

    def test_container_start_command_survives_ui_exit(self):
        cmd = pod.container_start_command()
        self.assertIn("/start.sh", cmd)        # still boots SSH + UI
        self.assertIn("|| true", cmd)           # UI exit doesn't fail the shell
        self.assertIn("sleep infinity", cmd)    # keep-alive holds the container

    def test_stop_key_env_prefers_scoped_key(self):
        with mock.patch.dict(os.environ, {"RUNPOD_STOP_API_KEY": "scoped",
                                          "RUNPOD_API_KEY": "account"}):
            self.assertEqual(pod.stop_key_env(), {"RUNPOD_STOP_KEY": "scoped"})

    def test_stop_key_env_falls_back_to_account_key_with_warning(self):
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "account_key"}, clear=True):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                env = pod.stop_key_env()
        self.assertEqual(env, {"RUNPOD_STOP_KEY": "account_key"})
        self.assertIn("WARNING", stderr.getvalue())

    def test_stop_key_env_empty_when_no_keys(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                env = pod.stop_key_env()
        self.assertEqual(env, {})
        self.assertIn("WARNING", stderr.getvalue())

    def test_gpu_aliases_resolve(self):
        self.assertEqual(pod.resolve_gpu_type("A100"), "NVIDIA A100 80GB PCIe")
        self.assertEqual(pod.resolve_gpu_type("A100 80GB"), "NVIDIA A100 80GB PCIe")
        self.assertEqual(pod.resolve_gpu_type("h100"), "NVIDIA H100 80GB HBM3")
        self.assertEqual(pod.resolve_gpu_type("4090"), "NVIDIA GeForce RTX 4090")
        self.assertEqual(pod.resolve_gpu_type("5090"), "NVIDIA GeForce RTX 5090")
        # exact RunPod ids pass through unchanged
        self.assertEqual(pod.resolve_gpu_type("NVIDIA L40S"), "NVIDIA L40S")


class TestReadPublicKey(unittest.TestCase):
    def setUp(self):
        self.ssh_dir = tempfile.mkdtemp(prefix="aitk-pod-test-ssh-")
        self.addCleanup(shutil.rmtree, self.ssh_dir, True)

    def _write(self, name, content):
        with open(os.path.join(self.ssh_dir, name), "w") as f:
            f.write(content + "\n")

    def test_prefers_ed25519_over_rsa(self):
        self._write("id_rsa.pub", "ssh-rsa AAAARSA u@m")
        self._write("id_ed25519.pub", "ssh-ed25519 AAAAED u@m")
        self.assertEqual(pod.read_public_key(self.ssh_dir), "ssh-ed25519 AAAAED u@m")

    def test_falls_back_to_rsa(self):
        self._write("id_rsa.pub", "ssh-rsa AAAARSA u@m")
        self.assertEqual(pod.read_public_key(self.ssh_dir), "ssh-rsa AAAARSA u@m")

    def test_missing_keys_raise_actionable_error(self):
        with self.assertRaises(pod.PodError) as ctx:
            pod.read_public_key(self.ssh_dir)
        self.assertIn("ssh-keygen", str(ctx.exception))


class TestSdkImport(unittest.TestCase):
    def test_missing_sdk_names_install_command(self):
        with mock.patch.dict(sys.modules, {"runpod": None}):
            with self.assertRaises(pod.PodError) as ctx:
                pod._get_sdk()
        self.assertIn("pip install -r scripts/remote/requirements.txt",
                      str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
