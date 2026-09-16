import unittest
import torch
from toolkit.scheduler import get_lr_scheduler


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.param = torch.nn.Parameter(torch.zeros(1))

    def _get_optimizer(self, lr=1e-3):
        return torch.optim.AdamW([self.param], lr=lr)

    def test_cosine_with_warmup_basic(self):
        opt = self._get_optimizer(lr=1e-3)
        sched = get_lr_scheduler('cosine_with_warmup', opt, total_iters=1000, num_warmup_steps=100)

        # Initial LR before steps
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.0)

        # Step through warmup
        for _ in range(100):
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 1e-3, places=6)

        # Step to halfway
        for _ in range(400):
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 1e-3 * 0.586824, places=4)

        # Step to end
        for _ in range(500):
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.0, places=6)

    def test_cosine_with_warmup_default_warmup(self):
        opt = self._get_optimizer(lr=1e-3)
        sched = get_lr_scheduler('cosine_with_warmup', opt, total_iters=1000)
        self.assertIsNotNone(sched)

    def test_cosine_with_hard_restarts(self):
        opt = self._get_optimizer(lr=1e-3)
        sched = get_lr_scheduler('cosine_with_hard_restarts_with_warmup', opt, total_iters=1000, num_warmup_steps=100, num_cycles=2)
        self.assertIsNotNone(sched)

    def test_all_schedulers_instantiation(self):
        schedulers = [
            'cosine',
            'constant',
            'linear',
            'step',
            'constant_with_warmup',
            'cosine_with_warmup',
            'cosine_with_hard_restarts_with_warmup',
        ]
        for name in schedulers:
            opt = self._get_optimizer(lr=1e-3)
            sched = get_lr_scheduler(name, opt, total_iters=100)
            self.assertIsNotNone(sched)


if __name__ == '__main__':
    unittest.main()
