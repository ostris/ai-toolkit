import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from toolkit.trigger_binding_artifacts import (
    ARTIFACT_TYPES,
    ArtifactValidationError,
    artifact_reference,
    build_checkpoint_manifest,
    canonical_json_dumps,
    config_fingerprint,
    decode_rng_state,
    encode_rng_state,
    load_artifact,
    load_checkpoint_manifest,
    phase_fingerprint,
    save_artifact,
    save_checkpoint_manifest,
    sha256_bytes,
    tensor_sha256,
    source_fingerprint,
)


class TriggerBindingArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.phase = {"name": "a1", "step": 12}
        self.source = {"model": "ideogram-4", "revision": "abc123"}
        self.config = {"rank": 1, "dtype": "bf16"}
        self.tensors = {
            "adapter.down.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "adapter.up.weight": torch.ones(3, 2, dtype=torch.bfloat16),
        }

    def _save(self, directory, artifact_type="te_adapter", tensors=None):
        path = Path(directory) / f"{artifact_type}.safetensors"
        manifest = save_artifact(
            path,
            artifact_type,
            tensors or self.tensors,
            phase=self.phase,
            source=self.source,
            config=self.config,
            extra={"trigger": "<literal>"},
        )
        return path, manifest

    def test_all_artifact_types_round_trip_with_metadata_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for artifact_type in sorted(ARTIFACT_TYPES):
                with self.subTest(artifact_type=artifact_type):
                    path, manifest = self._save(temp_dir, artifact_type)
                    loaded, loaded_manifest = load_artifact(
                        path,
                        expected_type=artifact_type,
                        expected_keys=self.tensors.keys(),
                        expected_shapes={key: tensor.shape for key, tensor in self.tensors.items()},
                        expected_phase_fingerprint=phase_fingerprint(self.phase),
                        expected_source_fingerprint=source_fingerprint(self.source),
                        expected_config_fingerprint=config_fingerprint(self.config),
                    )
                    self.assertEqual(manifest, loaded_manifest)
                    for key, tensor in self.tensors.items():
                        self.assertTrue(torch.equal(loaded[key], tensor))
                        self.assertEqual(
                            len(loaded_manifest["tensors"][key]["sha256"]),
                            64,
                        )

    def test_tensor_hash_supports_bfloat16_scalars_empty_and_noncontiguous_tensors(self):
        scalar = torch.tensor(1.25, dtype=torch.bfloat16)
        vector = scalar.reshape(1)
        self.assertEqual(tensor_sha256(scalar), tensor_sha256(vector))

        empty = torch.empty(0, dtype=torch.bfloat16)
        self.assertEqual(tensor_sha256(empty), sha256_bytes(b""))

        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        noncontiguous = base.transpose(0, 1)
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(
            tensor_sha256(noncontiguous),
            tensor_sha256(noncontiguous.contiguous()),
        )

    def test_artifact_round_trip_supports_bfloat16_scalar_tensor(self):
        tensors = {
            "adapter.scale": torch.tensor(1.0, dtype=torch.bfloat16),
            "adapter.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path, manifest = self._save(temp_dir, tensors=tensors)
            loaded, loaded_manifest = load_artifact(path, expected_type="te_adapter")
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual(loaded["adapter.scale"].shape, torch.Size([]))
            self.assertTrue(torch.equal(loaded["adapter.scale"], tensors["adapter.scale"]))

    def test_fingerprints_are_canonical_and_order_independent(self):
        self.assertEqual(config_fingerprint({"a": 1, "b": 2}), config_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(config_fingerprint({"a": 1}), config_fingerprint({"a": 2}))

    def test_load_fails_closed_on_expected_key_shape_and_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _ = self._save(temp_dir)
            cases = [
                {"expected_keys": ["adapter.down.weight"]},
                {"expected_shapes": {key: [99] for key in self.tensors}},
                {"expected_phase_fingerprint": "0" * 64},
                {"expected_source_fingerprint": "1" * 64},
                {"expected_config_fingerprint": "2" * 64},
                {"expected_file_sha256": "3" * 64},
                {"expected_type": "embedding"},
            ]
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ArtifactValidationError):
                    load_artifact(path, **kwargs)

    def test_load_rejects_unmanaged_safetensors_without_required_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bare.safetensors"
            save_file({"weight": torch.ones(1)}, str(path))
            with self.assertRaises(ArtifactValidationError):
                load_artifact(path)

    def test_load_rejects_tampered_tensor_manifest_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, manifest = self._save(temp_dir)
            tampered = json.loads(json.dumps(manifest))
            tampered["tensors"]["adapter.down.weight"]["sha256"] = "0" * 64
            manifest_json = canonical_json_dumps(tampered)
            metadata = {
                "trigger_binding.schema": tampered["schema"],
                "trigger_binding.schema_version": str(tampered["schema_version"]),
                "trigger_binding.artifact_type": tampered["artifact_type"],
                "trigger_binding.manifest": manifest_json,
                "trigger_binding.manifest_sha256": sha256_bytes(manifest_json.encode("utf-8")),
            }
            save_file(self.tensors, str(path), metadata=metadata)
            with self.assertRaisesRegex(ArtifactValidationError, "SHA-256 mismatch"):
                load_artifact(path)

    def test_atomic_save_preserves_previous_destination_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "te_adapter.safetensors"
            path.write_bytes(b"previous-good-file")
            with mock.patch("toolkit.trigger_binding_artifacts.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    save_artifact(
                        path,
                        "te_adapter",
                        self.tensors,
                        phase=self.phase,
                        source=self.source,
                        config=self.config,
                    )
            self.assertEqual(path.read_bytes(), b"previous-good-file")
            leftovers = [item for item in os.listdir(temp_dir) if item != path.name]
            self.assertEqual(leftovers, [])

    def test_rng_state_json_round_trip_supports_python_numpy_and_torch(self):
        state = {
            "python": random.Random(7).getstate(),
            "torch": torch.get_rng_state(),
            "cuda": [torch.arange(8, dtype=torch.uint8)],
        }
        try:
            import numpy as np

            state["numpy"] = np.random.RandomState(11).get_state()
        except ImportError:
            np = None

        encoded = encode_rng_state(state)
        json.dumps(encoded, allow_nan=False)
        decoded = decode_rng_state(encoded)
        self.assertEqual(decoded["python"], state["python"])
        self.assertTrue(torch.equal(decoded["torch"], state["torch"]))
        self.assertTrue(torch.equal(decoded["cuda"][0], state["cuda"][0]))
        if np is not None:
            self.assertEqual(decoded["numpy"][0], state["numpy"][0])
            self.assertTrue(np.array_equal(decoded["numpy"][1], state["numpy"][1]))
            self.assertEqual(decoded["numpy"][2:], state["numpy"][2:])

    def test_checkpoint_manifest_round_trip_and_artifact_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path, _ = self._save(temp_dir, "embedding")
            reference = artifact_reference(artifact_path, relative_to=temp_dir)
            manifest = build_checkpoint_manifest(
                phase=self.phase,
                source=self.source,
                config=self.config,
                step=42,
                artifacts={"embedding": reference},
                rng_state={"python": random.Random(3).getstate(), "torch": torch.get_rng_state()},
                extra={"optimizer": "adamw"},
            )
            manifest_path = Path(temp_dir) / "checkpoint_manifest.json"
            save_checkpoint_manifest(manifest_path, manifest)
            loaded = load_checkpoint_manifest(
                manifest_path,
                expected_phase_fingerprint=phase_fingerprint(self.phase),
                expected_source_fingerprint=source_fingerprint(self.source),
                expected_config_fingerprint=config_fingerprint(self.config),
            )
            self.assertEqual(loaded, manifest)
            decoded_rng = decode_rng_state(loaded["rng_state"])
            self.assertTrue(torch.equal(decoded_rng["torch"], torch.get_rng_state()))

            with open(artifact_path, "ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(ArtifactValidationError):
                load_checkpoint_manifest(manifest_path)

    def test_checkpoint_manifest_rejects_unknown_schema_keys(self):
        manifest = build_checkpoint_manifest(
            phase=self.phase,
            source=self.source,
            config=self.config,
            step=0,
            artifacts={},
            rng_state=None,
        )
        manifest["unexpected"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ArtifactValidationError):
                load_checkpoint_manifest(path, verify_artifacts=False)


if __name__ == "__main__":
    unittest.main()
