#!/usr/bin/env python3
"""Fail-closed launcher for supervised MoonViT-cache + Flow training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.runtime import require_idle_gpu


EXPECTED_SOURCE_SHA256 = "a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f"
EXPECTED_MODEL_PARAMETERS = 16_933_968
EXPECTED_TRAINABLE_PARAMETERS = 16_933_968
EXPECTED_FLOW_TRAINABLE_PARAMETERS = 13_687_632
EXPECTED_REPLACER_TRAINABLE_PARAMETERS = 3_246_336
EXPECTED_HA_SMOE_MISSING_PREFIXES = [
    "net.gcn.denoiser._contour_moe_router.",
    "net.gcn.denoiser.dit_layers.1.routed_moe.",
    "net.gcn.denoiser.dit_layers.3.routed_moe.",
    "net.gcn.denoiser.dit_layers.5.routed_moe.",
]
EXPECTED_BATCH_SIZE = 48
EXPECTED_MAX_STEPS = 60_000
EXPECTED_TRAIN_ROWS = 13_261
EXPECTED_TRAIN_CASES = 72
EXPECTED_DEV_ROWS = 1_123
EXPECTED_DEV_CASES = 8
EXPECTED_LOCAL_MILESTONES = [5_000, 10_000, 20_000, 40_000, 60_000]
EXPECTED_JITTER_PROBABILITIES = [0.35, 0.40, 0.20, 0.05]
EXPECTED_JITTER_SHIFT = [0.0, 0.05, 0.10, 0.15]
EXPECTED_JITTER_SCALE = [0.0, 0.10, 0.20, 0.30]
EXPECTED_JITTER_EDGE = [0.0, 0.03, 0.08, 0.15]
SOURCE_CHECKPOINT_STEP = 19_000
SOURCE_EQUIVALENT_ABSOLUTE_STEP = 59_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--config", type=Path, default=project_root / "configs/stage1.yaml")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--slice-manifest", type=Path, required=True)
    parser.add_argument("--moonvit-cache", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--preflight-steps", type=int, default=2)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--preflight-output",
        type=Path,
        help="completed preflight directory required by formal training",
    )
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def assert_float_list(actual, expected, label: str) -> None:
    observed = [float(value) for value in actual]
    if len(observed) != len(expected) or any(
        abs(left - right) > 1e-12 for left, right in zip(observed, expected)
    ):
        raise RuntimeError(f"{label} mismatch: {observed!r} != {expected!r}")


def extract_model_state(payload):
    for key in ("net", "model", "state_dict", "network"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate, key
    raise RuntimeError(f"checkpoint has no recognized model state: {sorted(payload.keys())}")


def module_for_name(name: str) -> str:
    normalized = name[7:] if name.startswith("module.") else name
    components = normalized.split(".")
    if "gcn" in components:
        return "flow"
    if "locate_feat_replacer" in components:
        return "moonvit_replacer"
    return "other"


def compare_preflight_states(source_path: Path, final_path: Path, audit_path: Path) -> None:
    import torch

    source_payload = torch.load(str(source_path), map_location="cpu", weights_only=False)
    final_payload = torch.load(str(final_path), map_location="cpu", weights_only=False)
    source_state, source_key = extract_model_state(source_payload)
    final_state, final_key = extract_model_state(final_payload)
    source_names = set(source_state)
    final_names = set(final_state)
    missing = sorted(source_names - final_names)
    added = sorted(final_names - source_names)
    disallowed_added = [
        name
        for name in added
        if not any(name.startswith(prefix) for prefix in EXPECTED_HA_SMOE_MISSING_PREFIXES)
    ]
    shape_mismatches = []
    stats = {
        name: {
            "tensor_count": 0,
            "changed_tensor_count": 0,
            "update_l2_squared": 0.0,
            "update_absmax": 0.0,
            "all_final_finite": True,
        }
        for name in ("flow", "moonvit_replacer", "other")
    }
    for name in sorted(source_names & final_names):
        before = source_state[name]
        after = final_state[name]
        if tuple(before.shape) != tuple(after.shape):
            shape_mismatches.append(
                {"name": name, "source": list(before.shape), "final": list(after.shape)}
            )
            continue
        group = module_for_name(name)
        group_stats = stats[group]
        group_stats["tensor_count"] += 1
        if after.is_floating_point() and not bool(torch.isfinite(after).all().item()):
            group_stats["all_final_finite"] = False
        delta = after.detach().to(dtype=torch.float64) - before.detach().to(dtype=torch.float64)
        delta_l2_squared = float(torch.sum(delta * delta).item())
        delta_absmax = float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
        group_stats["update_l2_squared"] += delta_l2_squared
        group_stats["update_absmax"] = max(group_stats["update_absmax"], delta_absmax)
        if delta_absmax > 0.0:
            group_stats["changed_tensor_count"] += 1

    for group_stats in stats.values():
        group_stats["update_l2"] = math.sqrt(group_stats.pop("update_l2_squared"))

    added_finite = all(
        not final_state[name].is_floating_point()
        or bool(torch.isfinite(final_state[name]).all().item())
        for name in added
    )
    final_step = int(final_payload.get("step", final_payload.get("global_step", -1)))
    passed = (
        not missing
        and bool(added)
        and not disallowed_added
        and added_finite
        and not shape_mismatches
        and final_step == 2
        and stats["flow"]["changed_tensor_count"] > 0
        and stats["moonvit_replacer"]["changed_tensor_count"] > 0
        and stats["other"]["tensor_count"] == 0
        and all(group["all_final_finite"] for group in stats.values())
    )
    audit = {
        "schema": "diffusionsnake.ha_smoe_supervised_preflight_audit.v1",
        "status": "PASS" if passed else "FAIL",
        "source_checkpoint": str(source_path),
        "final_checkpoint": str(final_path),
        "source_state_key": source_key,
        "final_state_key": final_key,
        "final_step": final_step,
        "missing_tensors": missing,
        "new_ha_smoe_tensors": added,
        "disallowed_new_tensors": disallowed_added,
        "all_new_ha_smoe_tensors_finite": added_finite,
        "shape_mismatches": shape_mismatches,
        "groups": stats,
        "interpretation": {
            "moonvit_encoder": "frozen offline cache; zero encoder parameters in training graph",
            "flow": "shared Flow and new HA-SMoE parameters are trainable",
            "moonvit_replacer": "must update",
        },
    }
    write_json(audit_path, audit)
    if not passed:
        raise RuntimeError(f"MoonViT Flow-tune preflight audit failed: {audit_path}")


def main():
    args = parse_args()
    worktree = Path(args.project_root).resolve()
    config = Path(args.config).resolve()
    source = Path(args.source_checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    slice_manifest = Path(args.slice_manifest).resolve()
    moonvit_cache = Path(args.moonvit_cache).resolve()
    output_root = Path(args.output_root).resolve()
    preflight_output = (
        Path(args.preflight_output).resolve()
        if args.preflight_output is not None else None
    )
    if not worktree.is_dir() or not config.is_file():
        raise FileNotFoundError({"worktree": str(worktree), "config": str(config)})
    if output_root.exists():
        raise RuntimeError(f"fresh output root required: {output_root}")
    for path, label, expect_directory in (
        (source, "source checkpoint", False),
        (data_root, "dataset root", True),
        (slice_manifest, "slice manifest", False),
        (moonvit_cache, "MoonViT cache", True),
    ):
        exists = path.is_dir() if expect_directory else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{label} missing: {path}")

    config_sha = sha256_file(config)
    assert_equal(
        sha256_file(source), EXPECTED_SOURCE_SHA256, "source checkpoint SHA256"
    )
    if args.mode == "train":
        if preflight_output is None:
            raise ValueError("--preflight-output is required for formal training")
        preflight_manifest_path = (
            preflight_output / "PURE2D_TRAINING_LAUNCH.json"
        )
        if not preflight_manifest_path.is_file():
            raise FileNotFoundError(
                f"completed preflight manifest missing: {preflight_manifest_path}"
            )
        preflight_manifest = json.loads(preflight_manifest_path.read_text())
        expected_identity = {
            "status": "COMPLETED",
            "mode": "preflight",
            "config_sha256": config_sha,
            "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
            "data_root": str(data_root),
            "slice_manifest": str(slice_manifest),
            "moonvit_cache": str(moonvit_cache),
        }
        observed_identity = {
            key: preflight_manifest.get(key) for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise RuntimeError(
                "preflight is incomplete or input identity changed: "
                f"{observed_identity!r} != {expected_identity!r}"
            )

    gpu_checks = require_idle_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["ONE_SAMPLE_RESUME_PATH"] = str(source)
    os.environ["DIFFUSIONSNAKE_DATA_ROOT"] = str(data_root)
    os.environ["DIFFUSIONSNAKE_SLICE_MANIFEST"] = str(slice_manifest)
    os.chdir(worktree)
    sys.path.insert(0, str(worktree))
    sys.argv = ["train.py", "--cfg_file", str(config)]
    os.environ["ONE_SAMPLE_OUT_DIR"] = str(output_root)

    import train

    cfg = train.cfg
    assert_equal(bool(cfg.resume), True, "resume")
    assert_equal(bool(cfg.resume_weights_only), True, "weights-only migration")
    assert_equal(bool(cfg.resume_allow_partial_copy), False, "partial-copy policy")
    assert_equal(
        [str(value) for value in cfg.resume_allowed_missing_prefixes],
        EXPECTED_HA_SMOE_MISSING_PREFIXES,
        "dense-to-HA-SMoE migration prefixes",
    )
    assert_equal(str(cfg.detector_backend), "flow_box_only", "detector backend")
    assert_equal(bool(cfg.use_gt_det_train_only), True, "GT train-only initialization")
    assert_equal(bool(cfg.locate_feat_inject), False, "Locate injection")
    assert_equal(bool(cfg.locate_feat_replace), True, "Locate replacement")
    assert_equal([str(value) for value in cfg.locate_feat_keys], ["layer_18"], "MoonViT keys")
    assert_equal(int(cfg.locate_feat_dim), 1_152, "MoonViT feature dimension")
    assert_equal(int(cfg.locate_feat_input_layers), 1, "MoonViT input layers")
    assert_equal(str(cfg.locate_feat_fusion_mode), "center_only", "MoonViT fusion")
    assert_equal(float(cfg.train.locate_lr_multiplier), 1.0, "replacer LR multiplier")
    assert_equal(int(cfg.pure2d_expected_parameter_count), EXPECTED_MODEL_PARAMETERS, "model parameters")
    assert_equal(int(cfg.train.batch_size), EXPECTED_BATCH_SIZE, "batch size")
    assert_equal(int(cfg.train.max_steps), EXPECTED_MAX_STEPS, "local max steps")
    assert_equal(int(cfg.train.warmup_steps), 1_000, "warmup steps")
    assert_equal(int(cfg.train.gradient_accumulation_steps), 1, "gradient accumulation")
    assert_equal(int(cfg.train.step_checkpoint_every), 1_000, "checkpoint interval")
    assert_equal(int(cfg.train.step_checkpoint_keep), 12, "checkpoint retention")
    assert_equal(
        [int(value) for value in cfg.train.step_checkpoint_milestones],
        EXPECTED_LOCAL_MILESTONES,
        "local checkpoint milestones",
    )
    assert_equal(int(cfg.sagittal_expected_train_row_count), EXPECTED_TRAIN_ROWS, "Train rows")
    assert_equal(int(cfg.sagittal_expected_train_case_count), EXPECTED_TRAIN_CASES, "Train cases")
    assert_equal(int(cfg.sagittal_expected_eval_row_count), EXPECTED_DEV_ROWS, "Dev rows")
    assert_equal(int(cfg.sagittal_expected_eval_case_count), EXPECTED_DEV_CASES, "Dev cases")
    assert_equal(str(cfg.diffusion_init_source), "bbox_octagon", "Route-B initialization")
    assert_equal(bool(cfg.routeb_box_jitter_enabled), True, "Route-B jitter")
    assert_float_list(cfg.routeb_box_jitter_probabilities, EXPECTED_JITTER_PROBABILITIES, "jitter probabilities")
    assert_float_list(cfg.routeb_box_jitter_shift_fractions, EXPECTED_JITTER_SHIFT, "jitter shift")
    assert_float_list(cfg.routeb_box_jitter_log_scale_fractions, EXPECTED_JITTER_SCALE, "jitter scale")
    assert_float_list(cfg.routeb_box_jitter_edge_fractions, EXPECTED_JITTER_EDGE, "jitter edge")
    assert_equal(float(cfg.routeb_box_jitter_min_iou), 0.20, "jitter IoU floor")

    cfg.resume_path = str(source)
    cfg.locate_feat_cache_root = str(moonvit_cache)
    import torch
    from lib.networks import make_network

    source_payload = torch.load(str(source), map_location="cpu", weights_only=False)
    source_step = int(source_payload.get("step", source_payload.get("global_step", -1)))
    assert_equal(source_step, SOURCE_CHECKPOINT_STEP, "source checkpoint step")
    del source_payload

    network = make_network(cfg)
    observed_total = sum(parameter.numel() for parameter in network.parameters())
    observed_trainable = sum(
        parameter.numel() for parameter in network.parameters() if parameter.requires_grad
    )
    flow_trainable = sum(
        parameter.numel() for parameter in network.gcn.parameters() if parameter.requires_grad
    )
    replacer_trainable = sum(
        parameter.numel()
        for parameter in network.locate_feat_replacer.parameters()
        if parameter.requires_grad
    )
    other_trainable = observed_trainable - flow_trainable - replacer_trainable
    assert_equal(observed_total, EXPECTED_MODEL_PARAMETERS, "observed model parameters")
    assert_equal(observed_trainable, EXPECTED_TRAINABLE_PARAMETERS, "observed trainable parameters")
    assert_equal(flow_trainable, EXPECTED_FLOW_TRAINABLE_PARAMETERS, "Flow trainable parameters")
    assert_equal(
        replacer_trainable,
        EXPECTED_REPLACER_TRAINABLE_PARAMETERS,
        "MoonViT replacer trainable parameters",
    )
    assert_equal(other_trainable, 0, "unexpected trainable parameters")
    del network

    formal_max_steps = int(cfg.train.max_steps)
    formal_num_workers = int(cfg.train.num_workers)
    formal_checkpoint_every = int(cfg.train.step_checkpoint_every)
    formal_checkpoint_keep = int(cfg.train.step_checkpoint_keep)
    if args.mode == "preflight":
        if args.preflight_steps != 2:
            raise ValueError("this signed preflight requires exactly 2 steps")
        cfg.train.max_steps = 2
        cfg.train.num_workers = 0
        cfg.train.step_checkpoint_every = 1
        cfg.train.step_checkpoint_keep = 2
        cfg.train.step_checkpoint_milestones = []

    output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "flowsnake.ha_smoe_stage1_from_moonvit_step19000.v1",
        "status": "STARTED",
        "mode": args.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "worktree": str(worktree),
        "config": str(config),
        "config_sha256": config_sha,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "source_checkpoint_step": SOURCE_CHECKPOINT_STEP,
        "source_equivalent_absolute_step": SOURCE_EQUIVALENT_ABSOLUTE_STEP,
        "data_root": str(data_root),
        "slice_manifest": str(slice_manifest),
        "moonvit_cache": str(moonvit_cache),
        "output_root": str(output_root),
        "physical_gpu": int(args.gpu),
        "gpu_idle_checks": gpu_checks,
        "preflight_output": (
            None if preflight_output is None else str(preflight_output)
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "local_max_steps": int(cfg.train.max_steps),
        "formal_local_max_steps": formal_max_steps,
        "batch_size": EXPECTED_BATCH_SIZE,
        "formal_num_workers": formal_num_workers,
        "formal_checkpoint_every": formal_checkpoint_every,
        "formal_checkpoint_keep": formal_checkpoint_keep,
        "local_milestones": EXPECTED_LOCAL_MILESTONES,
        "baseline_equivalent_absolute_steps": {
            str(step): SOURCE_EQUIVALENT_ABSOLUTE_STEP + step
            for step in EXPECTED_LOCAL_MILESTONES
        },
        "model_parameters": observed_total,
        "trainable_parameters": observed_trainable,
        "flow_trainable_parameters": flow_trainable,
        "moonvit_replacer_trainable_parameters": replacer_trainable,
        "moonvit_encoder_trainable_parameters": 0,
        "memory_parameters": 0,
        "internal_detector_parameters": 0,
        "moonvit_feature_source": "frozen_offline_layer18_cache",
        "moonvit_input_resolution": [448, 448],
        "moonvit_patch_size": 14,
        "moonvit_feature_dim": 1152,
        "moonvit_fusion_mode": "center_only",
        "frozen_modules": ["offline_moonvit_encoder"],
        "trainable_modules": [
            "shared_flow",
            "ha_smoe_contour_router",
            "ha_smoe_routed_experts",
            "moonvit_layer18_feature_replacer",
        ],
        "optimizer_transition": "fresh_adamw_dense_to_ha_smoe_weights_only_migration",
        "loss_role": "health_monitoring_only_no_plateau_early_stop",
    }
    manifest_path = output_root / "PURE2D_TRAINING_LAUNCH.json"
    write_json(manifest_path, manifest)
    try:
        train.main()
        if args.mode == "preflight":
            checkpoints = sorted(
                (output_root / "checkpoints").glob("step_*.pt"),
                key=lambda path: int(re.search(r"step_(\d+)\.pt$", path.name).group(1)),
            )
            if not checkpoints:
                raise RuntimeError("preflight produced no checkpoint")
            compare_preflight_states(
                source,
                checkpoints[-1],
                output_root / "FLOWTUNE_PREFLIGHT_AUDIT.json",
            )
    except BaseException as error:
        manifest["status"] = "FAILED"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        write_json(manifest_path, manifest)
        raise
    else:
        manifest["status"] = "COMPLETED"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
