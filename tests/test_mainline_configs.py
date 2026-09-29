from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class MainlineConfigTest(unittest.TestCase):
    def load(self, name):
        with (ROOT / "configs" / name).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_only_two_released_yaml_configs_exist(self):
        paths = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "configs").rglob("*.yaml")
        }
        self.assertEqual(
            paths,
            {"configs/stage1.yaml", "configs/stage2_rl.yaml"},
        )

    def test_only_one_diffusion_architecture_path_is_released(self):
        paths = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "lib" / "networks" / "diffusion").glob("*.py")
        }
        self.assertEqual(
            paths,
            {
                "lib/networks/diffusion/__init__.py",
                "lib/networks/diffusion/flow_matching_evolution.py",
                "lib/networks/diffusion/ha_smoe.py",
                "lib/networks/diffusion/mainline_denoiser.py",
            },
        )
        inference_source = (ROOT / "tools" / "infer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("EXPECTED_PARAMETERS = 16933968", inference_source)

    def test_stage1_contract(self):
        config = self.load("stage1.yaml")
        self.assertEqual(config["pure2d_expected_parameter_count"], 16_933_968)
        self.assertEqual(
            config["resume_path"],
            "artifacts/checkpoints/pure2d_moonvit_dense_step19000.pt",
        )
        self.assertEqual(
            config["resume_source_format"],
            "pure2d_moonvit_cached_flowtune_local_step19000_weights_only_v1",
        )
        self.assertEqual(
            config["resume_allowed_missing_prefixes"],
            [
                "net.gcn.denoiser._contour_moe_router.",
                "net.gcn.denoiser.dit_layers.1.routed_moe.",
                "net.gcn.denoiser.dit_layers.3.routed_moe.",
                "net.gcn.denoiser.dit_layers.5.routed_moe.",
            ],
        )
        self.assertEqual(config["train"]["max_steps"], 60_000)
        self.assertEqual(config["locate_feat_keys"], ["layer_18"])
        self.assertEqual(config["iterative_fractions"], [0.6667, 1.0])
        self.assertEqual(config["flow_ode_solver"], "ab2")

    def test_stage2_fourier_delta_nsd_contract(self):
        config = self.load("stage2_rl.yaml")
        self.assertEqual(config["pure2d_expected_parameter_count"], 16_933_968)
        self.assertEqual(config["rl_fractions"], [0.2, 0.25, 0.3333, 0.5, 1.0])
        self.assertEqual(config["rl_deployment_fractions"], [0.6667, 1.0])
        self.assertEqual(config["rl_deployment_ode_steps"], 4)
        self.assertEqual(config["rl_geom_lowfreq_modes"], 8)
        self.assertEqual(config["rl_geom_sigma_px"], [0.8, 0.7, 0.6, 0.5, 0.4])
        self.assertEqual(config["rl_per_step_credit_mode"], "full_extrap")
        self.assertTrue(config["rl_use_delta_nsd_reward"])
        self.assertEqual(config["rl_nsd_delta_px"], 2.0)
        self.assertTrue(config["rl_flow_only_update"])


if __name__ == "__main__":
    unittest.main()
