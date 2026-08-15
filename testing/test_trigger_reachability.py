import unittest

import torch
from torch import nn

from toolkit.trigger_reachability import (
    ReachabilityCheckError,
    check_gradient_reachability,
    check_optimizer_isolation,
    validate_reachability_and_isolation,
)


class _Activator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Parameter(torch.tensor([0.5, -0.25]))
        self.inactive_adapter = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

    def forward(self, value):
        return value * self.embedding.sum()


class _Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.5, 0.5]))

    def forward(self, value):
        return value * self.weight.sum()


def _configure_phase(activator, network, phase):
    activator.requires_grad_(phase in {"a1", "a2"})
    activator.inactive_adapter.requires_grad_(False)
    network.requires_grad_(phase == "b")


def _phase_forward(activator, network, phase, mode):
    value = torch.tensor([1.0, 2.0])
    if phase in {"a1", "a2"}:
        output = activator(value) if mode == "active" else value
    else:
        output = network(value) if mode == "active" else value
    return output


class TriggerReachabilityTest(unittest.TestCase):
    def _modules_and_optimizer(self, phase):
        activator = _Activator()
        network = _Network()
        _configure_phase(activator, network, phase)
        target = activator if phase in {"a1", "a2"} else network
        optimizer = torch.optim.SGD(
            [parameter for parameter in target.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        return activator, network, optimizer

    def test_static_gate_accepts_a1_a2_and_b_isolation(self):
        for phase in ("a1", "a2", "b"):
            with self.subTest(phase=phase):
                activator, network, optimizer = self._modules_and_optimizer(phase)
                diagnostics = check_optimizer_isolation(
                    activator,
                    network,
                    phase,
                    optimizer=optimizer,
                )
                self.assertTrue(diagnostics.passed, diagnostics.as_dict())
                self.assertTrue(diagnostics.complete)
                self.assertTrue(diagnostics.checks["target_in_optimizer"])
                self.assertTrue(diagnostics.checks["frozen_out_of_optimizer"])

    def test_static_gate_reports_pending_without_optimizer(self):
        activator, network, _ = self._modules_and_optimizer("a1")
        diagnostics = check_optimizer_isolation(activator, network, "a1")
        self.assertFalse(diagnostics.passed)
        self.assertFalse(diagnostics.complete)
        self.assertFalse(diagnostics.checks["optimizer_available"])
        self.assertIn("pending", " ".join(diagnostics.messages))

    def test_static_gate_rejects_frozen_parameter_in_optimizer(self):
        activator, network, _ = self._modules_and_optimizer("a1")
        optimizer = torch.optim.SGD(
            [activator.embedding, network.weight],
            lr=0.1,
        )
        diagnostics = check_optimizer_isolation(
            activator,
            network,
            "a1",
            optimizer=optimizer,
        )
        self.assertFalse(diagnostics.passed)
        self.assertFalse(diagnostics.checks["frozen_out_of_optimizer"])
        frozen = {item.name: item for item in diagnostics.parameters["frozen"]}
        self.assertTrue(frozen["network.weight"].in_optimizer)

    def test_gradient_gate_uses_scalar_loss_and_preserves_grad_fields(self):
        for phase in ("a1", "a2", "b"):
            with self.subTest(phase=phase):
                activator, network, optimizer = self._modules_and_optimizer(phase)
                active = _phase_forward(activator, network, phase, "active")
                bypass = _phase_forward(activator, network, phase, "bypass")
                loss = active.square().mean()
                diagnostics = check_gradient_reachability(
                    activator,
                    network,
                    phase,
                    optimizer=optimizer,
                    loss=loss,
                    active_output=active,
                    bypass_output=bypass,
                )
                self.assertTrue(diagnostics.passed, diagnostics.as_dict())
                self.assertGreater(diagnostics.output_difference, 0.0)
                self.assertTrue(all(item.grad_state == "nonzero" for item in diagnostics.parameters["target"]))
                self.assertTrue(all(parameter.grad is None for parameter in activator.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in network.parameters()))

    def test_forward_callback_runs_active_and_bypass_modes(self):
        activator, network, optimizer = self._modules_and_optimizer("a1")
        modes = []

        def forward(mode):
            modes.append(mode)
            output = _phase_forward(activator, network, "a1", mode)
            return {"output": output, "loss": output.square().mean() if mode == "active" else None}

        diagnostics = check_gradient_reachability(
            activator,
            network,
            "a1",
            optimizer=optimizer,
            forward_callback=forward,
        )
        self.assertTrue(diagnostics.passed, diagnostics.as_dict())
        self.assertEqual(modes, ["active", "bypass"])

    def test_gradient_gate_detects_zero_gradient_and_equal_outputs(self):
        activator, network, optimizer = self._modules_and_optimizer("b")
        active = network(torch.zeros(2))
        loss = active.sum()
        diagnostics = check_gradient_reachability(
            activator,
            network,
            "b",
            optimizer=optimizer,
            loss=loss,
            active_output=active,
            bypass_output=torch.zeros_like(active),
        )
        self.assertFalse(diagnostics.passed)
        self.assertFalse(diagnostics.checks["target_has_nonzero_gradient"])
        self.assertFalse(diagnostics.checks["active_bypass_outputs_differ"])

    def test_gradient_gate_detects_stale_frozen_grad(self):
        activator, network, optimizer = self._modules_and_optimizer("a2")
        network.weight.grad = torch.ones_like(network.weight)
        active = activator(torch.ones(2))
        diagnostics = check_gradient_reachability(
            activator,
            network,
            "a2",
            optimizer=optimizer,
            loss=active.sum(),
            active_output=active,
            bypass_output=torch.ones_like(active),
        )
        self.assertFalse(diagnostics.passed)
        self.assertFalse(diagnostics.checks["frozen_grad_absent"])

    def test_combined_api_is_incomplete_until_real_loss(self):
        activator, network, optimizer = self._modules_and_optimizer("a1")
        static_only = validate_reachability_and_isolation(
            activator,
            network,
            "a1",
            optimizer=optimizer,
        )
        self.assertFalse(static_only.complete)
        self.assertFalse(static_only.passed)
        self.assertTrue(static_only.static.passed)
        self.assertIsNone(static_only.gradient)

        active = activator(torch.ones(2))
        complete = validate_reachability_and_isolation(
            activator,
            network,
            "a1",
            optimizer=optimizer,
            loss=active.sum(),
            active_output=active,
            bypass_output=torch.ones_like(active),
        )
        self.assertTrue(complete.complete)
        self.assertTrue(complete.passed, complete.as_dict())

    def test_invalid_phase_and_non_scalar_loss_are_rejected(self):
        activator, network, optimizer = self._modules_and_optimizer("a1")
        with self.assertRaises(ReachabilityCheckError):
            check_optimizer_isolation(activator, network, "c", optimizer=optimizer)
        with self.assertRaisesRegex(ReachabilityCheckError, "scalar"):
            check_gradient_reachability(
                activator,
                network,
                "a1",
                optimizer=optimizer,
                loss=torch.ones(2, requires_grad=True),
                active_output=torch.ones(2),
                bypass_output=torch.zeros(2),
            )


if __name__ == "__main__":
    unittest.main()
