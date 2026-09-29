import math
import os
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("DIFFUSIONSNAKE_FULL_SMOKE") == "1",
    "set DIFFUSIONSNAKE_FULL_SMOKE=1 for the CPU model smoke test",
)
class MainlineModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_argv = sys.argv
        sys.argv = [
            original_argv[0],
            "--cfg_file",
            str(ROOT / "configs" / "stage1.yaml"),
        ]
        try:
            from lib.config import cfg
            from lib.networks import make_network
            from lib.rl.fourier import outer_action_mean, stage_progress
            from lib.train.trainers.diffusion_trainer import (
                DiffusionPretrainNetworkWrapper,
            )
            from lib.utils.snake.snake_gcn_utils import img_poly_to_can_poly
        finally:
            sys.argv = original_argv
        cls.cfg = cfg
        cls.make_network = staticmethod(make_network)
        cls.wrapper_type = DiffusionPretrainNetworkWrapper
        cls.canonicalize = staticmethod(img_poly_to_can_poly)
        cls.outer_action_mean = staticmethod(outer_action_mean)
        cls.stage_progress = staticmethod(stage_progress)

    @staticmethod
    def polygon(points, radius, center=64.0):
        angle = (
            torch.arange(points, dtype=torch.float32)
            * (2.0 * math.pi / float(points))
        )
        return torch.stack(
            (center + radius * angle.cos(), center + radius * angle.sin()),
            dim=-1,
        )

    def synthetic_batch(self):
        initial_40 = self.polygon(40, 28.0)
        initial_128 = self.polygon(128, 28.0)
        target_128 = self.polygon(128, 30.0)
        target_4 = torch.tensor(
            [[34.0, 34.0], [94.0, 34.0], [94.0, 94.0], [34.0, 94.0]],
            dtype=torch.float32,
        )

        def packed(value):
            return value.unsqueeze(0).unsqueeze(0)

        return {
            "inp": torch.zeros(1, 3, 512, 512),
            "locate_feat": [
                torch.randn(1152, 32, 32, dtype=torch.float32) * 0.01
            ],
            "locate_feat_scale": torch.tensor([[3.5]], dtype=torch.float32),
            "locate_feat_grid_hw": torch.tensor([[32, 32]], dtype=torch.int64),
            "locate_feat_patch_size": torch.tensor([[14]], dtype=torch.int64),
            "locate_feat_pad": torch.tensor(
                [[0, 0, 0, 0]], dtype=torch.int64
            ),
            "ct_01": torch.ones(1, 1),
            "ct_cls": torch.ones(1, 1, dtype=torch.int64),
            "ct_ind": torch.tensor([[64 * 128 + 64]], dtype=torch.int64),
            "wh": torch.tensor([[[60.0, 60.0]]]),
            "i_it_4py": packed(initial_40),
            "c_it_4py": packed(self.canonicalize(initial_40.unsqueeze(0))[0]),
            "i_gt_4py": packed(target_4),
            "c_gt_4py": packed(self.canonicalize(target_4.unsqueeze(0))[0]),
            "i_it_py": packed(initial_128),
            "c_it_py": packed(self.canonicalize(initial_128.unsqueeze(0))[0]),
            "i_gt_py": packed(target_128),
            "c_gt_py": packed(self.canonicalize(target_128.unsqueeze(0))[0]),
            "meta": {
                "ct_num": torch.tensor([1], dtype=torch.int64),
                "inv_trans_input": torch.tensor(
                    [[[0.25, 0.0, 0.0], [0.0, 0.25, 0.0]]],
                    dtype=torch.float32,
                ),
                "orig_hw": torch.tensor([[128.0, 128.0]]),
                "flipped": torch.zeros(1, 1),
            },
        }

    def test_supervised_backward_and_deployed_inference(self):
        torch.manual_seed(20260824)
        network = self.make_network(self.cfg)
        wrapper = self.wrapper_type(network)
        self.assertEqual(
            sum(parameter.numel() for parameter in wrapper.parameters()),
            16_933_968,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in network.gcn.parameters()),
            13_687_632,
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in network.locate_feat_replacer.parameters()
            ),
            3_246_336,
        )

        batch = self.synthetic_batch()
        wrapper.train()
        _, loss, statistics, _ = wrapper(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and parameter.grad.abs().sum() > 0
                for parameter in network.gcn.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and parameter.grad.abs().sum() > 0
                for parameter in network.locate_feat_replacer.parameters()
            )
        )
        self.assertIn("diff_loss", statistics)

        self.cfg.use_gt_det_train_only = False
        network.eval()
        with torch.no_grad():
            prediction = network(batch["inp"], batch)
        self.assertEqual(tuple(prediction["py"].shape), (1, 128, 2))
        self.assertTrue(torch.isfinite(prediction["py"]).all())
        self.assertEqual(tuple(prediction["detection"].shape), (1, 1, 6))

        feature = prediction["cnn_feature"]
        initial = prediction["i_it_py"]
        owner = prediction["py_ind"]

        def helper_displacement(fractions):
            progress = self.stage_progress(fractions)
            latents = [torch.randn_like(initial) for _ in fractions]
            current = initial.clone()
            total = torch.zeros_like(initial)
            for index, fraction in enumerate(fractions):
                canonical = self.canonicalize(current)
                action = self.outer_action_mean(
                    network.gcn,
                    feature,
                    current,
                    canonical,
                    owner,
                    fraction,
                    4,
                    progress[index],
                    latents[index],
                )
                current = current + action
                total = total + action
            return total

        for fractions in (
            (0.6667, 1.0),
            (0.2, 0.25, 0.3333, 0.5, 1.0),
        ):
            random_state = torch.get_rng_state()
            production = network.gcn.sample_disp_iterative(
                feature,
                initial,
                self.canonicalize(initial),
                owner,
                num_iter_steps=len(fractions),
                fractions=fractions,
                ode_steps=4,
            )
            torch.set_rng_state(random_state)
            helper = helper_displacement(fractions)
            self.assertLessEqual(
                float((production - helper).abs().max().item()),
                1e-5,
            )


if __name__ == "__main__":
    unittest.main()
