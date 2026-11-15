"""Offline MOSS adapter serialization regressions; no model weights are loaded."""

import contextlib
import io
import tempfile
import unittest
import weakref
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from toolkit.models.lokr import LokrModule
from toolkit.network_mixins import ToolkitNetworkMixin


def moss_model():
    # Conversion only needs the prefix convention, not a loaded denoiser/encoder.
    model = object.__new__(MossImageModel)
    model.lora_keys_use_comfy_prefix = True
    return model


class TinyNetwork(ToolkitNetworkMixin, torch.nn.Module):
    """Real LoKr module and network save/load path with one small projection."""

    name = "transformer$$layers$$0$$proj$$qkv"

    def __init__(self, model, rank, dtype):
        torch.nn.Module.__init__(self)
        ToolkitNetworkMixin.__init__(self)
        self.base_model_ref = weakref.ref(model)
        self.network_type = "lokr"
        self.peft_format = True
        self.is_pixart = False
        self.use_old_lokr_format = False
        self.add_module(
            self.name,
            LokrModule(
                self.name,
                torch.nn.Linear(8, 24, bias=False),
                lora_dim=rank,
                alpha=rank,
                factor=2,
                network=self,
            ).to(dtype=dtype),
        )
        self.load_result = None

    def get_keymap(self, force_weight_mapping=False):
        return {}

    def load_state_dict(self, *args, **kwargs):
        self.load_result = super().load_state_dict(*args, **kwargs)
        return self.load_result

    @property
    def adapter(self):
        return self.get_submodule(self.name)


class MossLokrSerializationTest(unittest.TestCase):
    def setUp(self):
        self.model = moss_model()

    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertEqual(actual[key].dtype, expected[key].dtype)
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_new_and_legacy_files_load_through_network_without_dropped_keys(self):
        torch.manual_seed(17)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for rank in (1, 64):
                with self.subTest(dtype=dtype, rank=rank):
                    source = TinyNetwork(self.model, rank, dtype)
                    with torch.no_grad():
                        for parameter in source.parameters():
                            parameter.normal_(std=0.1)
                    original = {k: v.clone() for k, v in source.state_dict().items()}
                    saved = source.get_state_dict(dtype=dtype)
                    prefix = "diffusion_model.layers.0.proj."
                    self.assertIn(prefix + "qkv.alpha", saved)
                    self.assertFalse(
                        any(prefix + p + ".alpha" in saved for p in ("q", "k", "v"))
                    )
                    legacy = dict(saved)
                    alpha = legacy.pop(prefix + "qkv.alpha")
                    legacy.update(
                        {prefix + p + ".alpha": alpha.clone() for p in ("q", "k", "v")}
                    )
                    for state in (saved, legacy):
                        with tempfile.TemporaryDirectory(
                            prefix="moss-lokr-"
                        ) as directory:
                            path = str(Path(directory) / "adapter.safetensors")
                            save_file(state, path, metadata={"test": "MOSS LoKr"})
                            target = TinyNetwork(self.model, rank, dtype)
                            output = io.StringIO()
                            with contextlib.redirect_stdout(output):
                                extra = target.load_weights(path)
                            self.assertIsNone(extra)
                            self.assertNotIn("forced keymap", output.getvalue())
                            self.assertEqual(target.load_result.missing_keys, [])
                            self.assertEqual(target.load_result.unexpected_keys, [])
                            self.assert_state_equal(target.state_dict(), original)
                            # Independent dense reconstruction of the learned delta.
                            adapter = target.adapter
                            w2 = (
                                adapter.lokr_w2
                                if adapter.use_w2
                                else adapter.lokr_w2_a.float()
                                @ adapter.lokr_w2_b.float()
                            )
                            expected = torch.kron(adapter.lokr_w1.float(), w2.float())
                            torch.testing.assert_close(
                                target.adapter.get_weight().float(),
                                expected,
                                rtol=0.02,
                                atol=0.002,
                            )
                            self.assert_state_equal(load_file(path), state)

    def test_legacy_42_block_checkpoint_preserves_all_payloads(self):
        legacy, expected = {}, {}
        for i in range(42):
            saved = f"diffusion_model.layers.{i}.proj."
            loaded = f"transformer.layers.{i}.proj."
            for suffix in ("lokr_w1", "lokr_w2_a", "lokr_w2_b"):
                value = torch.full((2, 2), i + 0.5, dtype=torch.bfloat16)
                legacy[saved + "qkv." + suffix] = value
                expected[loaded + "qkv." + suffix] = value
            alpha = torch.tensor(8, dtype=torch.bfloat16)
            for p in ("q", "k", "v"):
                legacy[saved + p + ".alpha"] = alpha.clone()
            expected[loaded + "qkv.alpha"] = alpha
        before = {k: v.clone() for k, v in legacy.items()}
        actual = self.model.convert_lora_weights_before_load(legacy)
        self.assert_state_equal(actual, expected)
        self.assert_state_equal(legacy, before)
        self.assert_state_equal(
            self.model.convert_lora_weights_before_load(
                self.model.convert_lora_weights_before_save(actual)
            ),
            expected,
        )

    def test_conflicting_legacy_or_fused_alpha_is_rejected_before_loading(self):
        for conflict in ("k", "qkv"):
            with self.subTest(conflict=conflict):
                prefix = "diffusion_model.layers.0.proj."
                state = {prefix + "qkv.lokr_w1": torch.ones(2, 2)}
                state.update(
                    {prefix + p + ".alpha": torch.tensor(4.0) for p in ("q", "k", "v")}
                )
                state[prefix + conflict + ".alpha"] = torch.tensor(8.0)
                before = {k: v.clone() for k, v in state.items()}
                target = TinyNetwork(self.model, 64, torch.float32)
                original = {k: v.clone() for k, v in target.state_dict().items()}
                with self.assertRaisesRegex(
                    ValueError, "Conflicting MOSS LoKr QKV alpha"
                ):
                    target.load_weights(state)
                self.assertIsNone(target.load_result)
                self.assert_state_equal(target.state_dict(), original)
                self.assert_state_equal(state, before)

    def test_duplicate_fused_partial_split_and_missing_alpha(self):
        prefix = "transformer.layers.0.proj."
        for projections in ((), ("q",), ("q", "k", "v"), ("qkv", "q", "k", "v")):
            with self.subTest(projections=projections):
                state = {prefix + "qkv.lokr_w1_a": torch.ones(2, 1)}
                for p in projections:
                    state[prefix + p + ".alpha"] = torch.tensor(2.0)
                expected = {prefix + "qkv.lokr_w1_a": state[prefix + "qkv.lokr_w1_a"]}
                if projections:
                    expected[prefix + "qkv.alpha"] = torch.tensor(2.0)
                self.assert_state_equal(
                    self.model.convert_lora_weights_before_load(state), expected
                )

    def test_unrelated_keys_are_preserved_and_not_mistaken_for_lokr(self):
        state = {
            "transformer.layers.0.proj.q.alpha": torch.tensor(3.0),
            "transformer.layers.0.proj.o.alpha": torch.tensor(4.0),
            "transformer.layers.0.proj.o.lokr_w1": torch.ones(2, 2),
        }
        self.assert_state_equal(
            self.model.convert_lora_weights_before_load(state), state
        )
        self.assert_state_equal(
            self.model.convert_lora_weights_before_load(
                self.model.convert_lora_weights_before_save(state)
            ),
            state,
        )

    def test_standard_lora_round_trip_keeps_shared_qkv_behavior(self):
        for down_suffix, up_suffix in (
            ("lora_A.weight", "lora_B.weight"),
            ("lora_down.weight", "lora_up.weight"),
        ):
            with self.subTest(down_suffix=down_suffix):
                prefix = "transformer.layers.0.proj.qkv."
                state = {
                    prefix + down_suffix: torch.arange(8.0).reshape(2, 4),
                    prefix + up_suffix: torch.arange(18.0).reshape(9, 2),
                    prefix + "alpha": torch.tensor(2.0),
                }
                saved = self.model.convert_lora_weights_before_save(state)
                self.assertNotIn("diffusion_model.layers.0.proj.qkv.alpha", saved)
                self.assert_state_equal(
                    self.model.convert_lora_weights_before_load(saved), state
                )

    def test_independent_standard_lora_projections_keep_effective_weights(self):
        state, deltas = {}, []
        prefix = "transformer.layers.0.proj."
        for i, p in enumerate(("q", "k", "v")):
            down = torch.arange(8.0).reshape(2, 4) + i
            up = torch.arange(6.0).reshape(3, 2) - i
            state[prefix + p + ".lora_A.weight"] = down
            state[prefix + p + ".lora_B.weight"] = up
            state[prefix + p + ".alpha"] = torch.tensor(4.0)
            deltas.append((up @ down) * 2)
        loaded = self.model.convert_lora_weights_before_load(state)
        down = loaded[prefix + "qkv.lora_A.weight"]
        up = loaded[prefix + "qkv.lora_B.weight"]
        scale = loaded[prefix + "qkv.alpha"] / down.shape[0]
        torch.testing.assert_close(
            (up @ down) * scale, torch.cat(deltas), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
