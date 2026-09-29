import unittest

import torch

from lib.networks.diffusion.ha_smoe import (
    ContourRoutePath,
    ContourSharedResidualExperts,
)
from lib.networks.diffusion.mainline_denoiser import (
    DenseResidualFinalHead,
    MainlineFlowDenoiser,
)


class HASMoEArchitectureTest(unittest.TestCase):
    def test_route_is_one_decision_per_contour_for_all_routed_blocks(self):
        torch.manual_seed(7)
        router = ContourRoutePath(
            dim=32,
            num_routed_blocks=3,
            num_experts=4,
            hidden_dim=48,
        )
        self.assertEqual(router.descriptor_dim, 64)
        logits = router(torch.randn(2, 16, 32))
        self.assertEqual(tuple(logits.shape), (2, 3, 4))
        self.assertNotIn(16, logits.shape)
        diagnostics = router.diagnostics()
        self.assertIn("route_entropy", diagnostics)
        self.assertTrue(torch.isfinite(diagnostics["route_entropy"]))

    def test_top2_choice_is_shared_by_all_points(self):
        torch.manual_seed(11)
        experts = ContourSharedResidualExperts(
            dim=16,
            num_experts=4,
            top_k=2,
            expert_hidden_dim=24,
        )
        tokens = torch.randn(2, 9, 16, requires_grad=True)
        logits = torch.tensor(
            [[8.0, 7.0, -3.0, -4.0], [-4.0, -3.0, 7.0, 8.0]]
        )
        output = experts(tokens, logits)
        self.assertEqual(tuple(output.shape), tuple(tokens.shape))
        loss = output.square().mean() + experts.reg_loss()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(int(experts.diagnostics()["dead_experts"].item()), 0)

    def test_mainline_contains_only_standard_ha_smoe_and_dense_head(self):
        model = MainlineFlowDenoiser()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            13_687_632,
        )
        self.assertEqual(model._ha_smoe_layer_indices, (1, 3, 5))
        routed = tuple(
            index
            for index, layer in enumerate(model.dit_layers)
            if layer.routed_moe is not None
        )
        self.assertEqual(routed, (1, 3, 5))
        self.assertIsInstance(model.final_layer, DenseResidualFinalHead)
        state_names = tuple(model.state_dict())
        self.assertTrue(
            any(name.startswith("_contour_moe_router.") for name in state_names)
        )
        self.assertFalse(any("final_layer.experts" in name for name in state_names))
        self.assertFalse(any("routed_moe.router" in name for name in state_names))

    def test_dense_checkpoint_is_an_explicit_subset_of_ha_smoe(self):
        model = MainlineFlowDenoiser()
        full_state = model.state_dict()
        new_prefixes = (
            "_contour_moe_router.",
            "dit_layers.1.routed_moe.",
            "dit_layers.3.routed_moe.",
            "dit_layers.5.routed_moe.",
        )
        dense_state = {
            name: value.clone()
            for name, value in full_state.items()
            if not name.startswith(new_prefixes)
        }
        incompatible = model.load_state_dict(dense_state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(name.startswith(new_prefixes) for name in incompatible.missing_keys)
        )

    def test_local_router_forward_is_finite_and_reports_three_blocks(self):
        torch.manual_seed(13)
        model = MainlineFlowDenoiser(
            state_dim=32,
            feature_dim=32,
            num_heads=4,
            num_points=16,
            num_queries=12,
            dense_residual_hidden_dim=64,
        )
        prediction, regularization = model(
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 32, 16),
            torch.randn(2, 16, 2),
            torch.tensor([100.0, 700.0]),
            py_ind=torch.tensor([0, 1]),
            s=torch.tensor([0.0, 0.6667]),
        )
        self.assertEqual(tuple(prediction.shape), (2, 16, 2))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(regularization))
        diagnostics = model.moe_diagnostics()
        self.assertTrue(any(key.startswith("contour.") for key in diagnostics))
        for block in (2, 4, 6):
            self.assertTrue(
                any(key.startswith(f"block{block}.") for key in diagnostics)
            )


if __name__ == "__main__":
    unittest.main()
