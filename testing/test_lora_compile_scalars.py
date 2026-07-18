import unittest
from unittest import mock

import torch

from toolkit.kohya_lora import LoRAModule as KohyaLoRAModule
from toolkit.lora_special import LoRAModule
from toolkit.lycoris_special import LoConSpecialModule
from toolkit.models.DoRA import DoRAModule
from toolkit.models.lokr import LokrModule


class _Network:
    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0


def _linear(device=None, dtype=None):
    return torch.nn.Linear(8, 8, bias=False, device=device, dtype=dtype)


class AdapterScaleTest(unittest.TestCase):
    def test_adapters_keep_float_metadata_and_nonpersistent_runtime_buffer(self):
        network = _Network()
        modules = [
            LoRAModule(
                "lora_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            KohyaLoRAModule(
                "kohya_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
            ),
            LoConSpecialModule(
                "locon_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            DoRAModule(
                "dora_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            LokrModule(
                "lokr_scale",
                _linear(),
                lora_dim=2,
                alpha=torch.tensor(4, dtype=torch.bfloat16),
                network=network,
            ),
        ]

        for module in modules:
            with self.subTest(module=type(module).__name__):
                self.assertIs(type(module.scale), float)
                self.assertEqual(module._runtime_scale.item(), module.scale)
                self.assertNotIn("_runtime_scale", module.state_dict())
                self.assertFalse(module._runtime_scale.requires_grad)

    def test_extract_weight_synchronizes_runtime_scale(self):
        module = LoRAModule(
            "extract_scale",
            _linear(),
            lora_dim=4,
            alpha=torch.tensor(8, dtype=torch.bfloat16),
            network=_Network(),
        )
        runtime_scale = module._runtime_scale
        down = torch.randn(2, 8)
        up = torch.randn(8, 2)

        with mock.patch(
            "toolkit.network_mixins.extract_linear",
            return_value=(down, up, 2, None),
        ):
            module.extract_weight(extract_mode="fixed", extract_mode_param=2)

        self.assertIs(module._runtime_scale, runtime_scale)
        self.assertEqual(module.scale, 1.0)
        self.assertEqual(module._runtime_scale.item(), 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dynamic_compile_stays_cuda_and_scale_updates_do_not_recompile(self):
        torch.manual_seed(0)
        torch._dynamo.reset()
        from torch._inductor import metrics

        metrics.reset()
        network = _Network()
        network.torch_multiplier = torch.ones(1, device="cuda")
        original = _linear(device="cuda", dtype=torch.bfloat16)
        original.requires_grad_(False)
        module = LoRAModule(
            "compiled_scale",
            original,
            lora_dim=4,
            alpha=torch.tensor(8, dtype=torch.bfloat16),
            network=network,
        ).to("cuda")
        module.org_forward = original.forward
        with torch.no_grad():
            module.lora_up.weight.normal_()

        value = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        eager = module(value)
        eager.square().mean().backward()
        eager_down_grad = module.lora_down.weight.grad.detach().clone()
        eager_up_grad = module.lora_up.weight.grad.detach().clone()
        module.zero_grad(set_to_none=True)

        compiled = torch.compile(module, fullgraph=False, dynamic=True)
        actual = compiled(value)
        actual.square().mean().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, eager, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(
            module.lora_down.weight.grad, eager_down_grad, rtol=2e-2, atol=5e-2
        )
        torch.testing.assert_close(
            module.lora_up.weight.grad, eager_up_grad, rtol=2e-2, atol=5e-2
        )
        self.assertEqual(
            getattr(metrics, "generated_cpp_vec_kernel_count", 0), 0
        )
        self.assertEqual(module._runtime_scale.device.type, "cuda")

        base = original(value)
        original_delta = actual - base
        generated_kernels = metrics.generated_kernel_count
        module._set_runtime_scale(0.5)
        updated = compiled(value)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            updated - base, original_delta * 0.25, rtol=2e-2, atol=5e-2
        )
        self.assertEqual(metrics.generated_kernel_count, generated_kernels)


if __name__ == "__main__":
    unittest.main()
