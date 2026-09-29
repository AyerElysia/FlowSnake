#!/usr/bin/env python3
"""Flow-only GRPO for the official five-action Fourier mainline.

Training uses five residual outer stages (20/40/60/80/100 percent cumulative
progress). Each stage owns one low-frequency Fourier normal-field action.
Every reported validation result uses the immutable production solver instead:
two outer stages with four deterministic AB2 evaluations each (8 NFE total).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# lib.config consumes --cfg_file while importing. Preserve it and hide any
# launcher-only arguments before importing the shared configuration parser.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cfg_file", default="", type=str)
_pre_args, _remaining_argv = _pre_parser.parse_known_args()
if _pre_args.cfg_file:
    os.environ["CFG_FILE"] = _pre_args.cfg_file
sys.argv = [sys.argv[0]]
if _pre_args.cfg_file:
    sys.argv += ["--cfg_file", _pre_args.cfg_file]
sys.argv += _remaining_argv

import numpy as np
import torch
from torch import nn

from lib.checkpoints import extract_state_dict, normalize_state_dict, sha256_file
from lib.config import args, cfg
from lib.datasets import make_data_loader
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.networks import make_network
from lib.rl import (
    fourier_action_logprob,
    fourier_mean_kl,
    low_frequency_delta,
    outer_action_mean,
    stage_progress,
    standard_normal_logprob,
)
from lib.train.grpo_v2_utils import EMA, freeze_bn_running_stats, percentiles
from lib.train.rewards.region_reward import (
    compute_delta_nsd_reward,
    compute_nsd_score,
    compute_region_score,
)
from lib.utils.snake import snake_config, snake_gcn_utils


EXPECTED_SOURCE_SHA256 = os.environ.get(
    "DIFFUSIONSNAKE_SOURCE_SHA256", ""
).strip().lower()
EXPECTED_SOURCE_STEP = int(os.environ.get("DIFFUSIONSNAKE_SOURCE_STEP", "-1"))
EXPECTED_MODEL_PARAMETERS = 16_933_968
EXPECTED_FLOW_PARAMETERS = 13_687_632
EXPECTED_CONTEXT_PARAMETERS = 3_246_336
EXPECTED_BACKEND = "flow_box_only"
EXPECTED_TRAIN_FRACTIONS = (0.2, 0.25, 0.3333, 0.5, 1.0)
EXPECTED_TRAIN_PROGRESS = (0.0, 0.2, 0.4, 0.6, 0.8)
EXPECTED_DEPLOYMENT_FRACTIONS = (0.6667, 1.0)
EXPECTED_DEPLOYMENT_PROGRESS = (0.0, 0.6667)
EXPECTED_SIGMA_PX = (0.8, 0.7, 0.6, 0.5, 0.4)
EXPECTED_NSD_DELTA_PX = 2.0
TUNE_DATASET = "VolMemFourierVal37"
TUNE_MANIFEST = PROJECT_ROOT / "configs/manifests/volmem_fourier_validation37.csv"
EXPECTED_TUNE_MANIFEST_SHA256 = (
    "24a4f19651edb5d187029f0255e2b59f9dce40f320ee29c14b709e8e92e6e6ad"
)
LOG_SUBDIR = "rl"


class _CheckpointWrapper(nn.Module):
    """Retain the supervised ``net.*`` checkpoint namespace."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.net = network


def _cfg_value(name: str, default):
    raw = (
        os.environ.get("RL_STEPS")
        if name == "train_steps" and "RL_STEPS" in os.environ
        else os.environ.get(f"RL_{name.upper()}")
    )
    if raw is not None:
        raw = str(raw).strip()
        if isinstance(default, bool):
            return raw.lower() in ("1", "true", "yes", "on")
        if isinstance(default, int) and not isinstance(default, bool):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, (tuple, list)):
            return [float(value) for value in raw.strip("[]").split(",") if value.strip()]
        return raw
    nested = getattr(cfg, "rl", None)
    if nested is not None and name in nested:
        return nested[name]
    return getattr(cfg, f"rl_{name}", default)


def _project_path(value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cfg_file_used() -> str:
    return str(getattr(args, "cfg_file", "") or os.environ.get("CFG_FILE", ""))


def _output_dir() -> Path:
    override = os.environ.get("RL_OUTPUT_DIR", "").strip()
    return _project_path(override or getattr(cfg, "model_dir", "data/outputs/stage2_rl"))


def _source_checkpoint() -> Path:
    candidates = (
        os.environ.get("CKPT_PATH", "").strip(),
        str(getattr(args, "ckpt", "") or "").strip(),
        str(getattr(cfg, "resume_path", "") or "").strip(),
    )
    for candidate in candidates:
        if candidate:
            path = _project_path(candidate)
            if path.is_file() and path.suffix in (".pt", ".pth"):
                return path
    raise FileNotFoundError("set CKPT_PATH or cfg.resume_path to the stage-1 checkpoint")


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Dict, device: str = "cuda") -> Dict:
    for key, value in list(batch.items()):
        if key != "meta" and isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def _flatten_valid_polys(batch: Dict, key: str, device) -> torch.Tensor:
    polys = batch.get(key)
    if not isinstance(polys, torch.Tensor):
        raise KeyError(f"batch missing tensor {key}")
    if polys.dim() == 4:
        valid = batch.get("ct_01")
        polys = polys[valid.bool()] if isinstance(valid, torch.Tensor) else polys.flatten(0, 1)
    elif polys.dim() != 3:
        raise ValueError(f"{key} must be 3D or 4D, got {tuple(polys.shape)}")
    return polys.to(device=device)


def _make_py_ind(batch: Dict, n_contours: int, device) -> torch.Tensor:
    valid = batch.get("ct_01")
    if isinstance(valid, torch.Tensor) and valid.dim() == 2:
        indices = [
            torch.full(
                (int(valid[index].sum().item()),),
                index,
                dtype=torch.long,
                device=device,
            )
            for index in range(valid.size(0))
        ]
        if indices:
            return torch.cat(indices)
    return torch.zeros(n_contours, dtype=torch.long, device=device)


def _signed_area(poly: torch.Tensor) -> torch.Tensor:
    x, y = poly[..., 0], poly[..., 1]
    return 0.5 * (x * torch.roll(y, -1, 1) - torch.roll(x, -1, 1) * y).sum(1)


def _align_gt(initial: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if initial.size(0) == 0:
        return target
    target = target.clone()
    reversed_winding = (_signed_area(initial) >= 0) ^ (_signed_area(target) >= 0)
    if reversed_winding.any():
        target[reversed_winding] = torch.flip(target[reversed_winding], dims=(1,))
    nearest = ((initial[:, :1] - target).square().sum(-1)).argmin(1)
    return torch.stack(
        [torch.roll(target[index], -int(nearest[index]), dims=0) for index in range(target.size(0))]
    )


def _burr_penalty(
    final_poly: torch.Tensor,
    gt_poly: torch.Tensor,
    coord_scale: float,
    margin_px: float,
    max_px: float,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    def laplacian(poly):
        poly = poly * float(coord_scale)
        return torch.roll(poly, 1, 1) - 2.0 * poly + torch.roll(poly, -1, 1)

    final_curvature = laplacian(final_poly).norm(dim=-1)
    target_curvature = laplacian(gt_poly).norm(dim=-1)
    excess = torch.relu(final_curvature - target_curvature - float(margin_px))
    raw = torch.quantile(excess, q=float(quantile) / 100.0, dim=1)
    return raw.div(max(float(max_px), 1e-6)).clamp(0.0, 2.0), raw


def _extract_wrapper_state(payload: dict, model: nn.Module) -> dict:
    state = normalize_state_dict(extract_state_dict(payload))
    if any(name.startswith("net.") for name in model.state_dict()) and not any(
        name.startswith("net.") for name in state
    ):
        state = {f"net.{name}": tensor for name, tensor in state.items()}
    return state


def _strict_load(model: nn.Module, payload: dict, label: str) -> dict:
    state = _extract_wrapper_state(payload, model)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape = sorted(
        name
        for name in set(expected) & set(state)
        if tuple(expected[name].shape) != tuple(state[name].shape)
    )
    nonfinite = sorted(
        name
        for name, tensor in state.items()
        if torch.is_tensor(tensor)
        and tensor.is_floating_point()
        and not torch.isfinite(tensor).all().item()
    )
    if missing or unexpected or shape or nonfinite:
        raise RuntimeError(
            f"{label} strict load failed: missing={missing[:8]} "
            f"unexpected={unexpected[:8]} shape={shape[:8]} nonfinite={nonfinite[:8]}"
        )
    model.load_state_dict(state, strict=True)
    return state


def _finite_scalar(value, name: str) -> float:
    value = float(value.detach().item()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite: {value}")
    return value


def _require_exact(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} drift: {actual!r} != {expected!r}")


def main() -> None:
    print("[RL] five Fourier actions for training; immutable AB2 2x4 deployment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RL training")

    cache_override = os.environ.get("MOONVIT_CACHE_ROOT", "").strip()
    if cache_override:
        cfg.locate_feat_cache_root = cache_override
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    train_steps = int(_cfg_value("train_steps", 10_000))
    k_rollouts = int(_cfg_value("k", 8))
    fractions = tuple(float(x) for x in _cfg_value("fractions", EXPECTED_TRAIN_FRACTIONS))
    deployment_fractions = tuple(
        float(x) for x in _cfg_value("deployment_fractions", EXPECTED_DEPLOYMENT_FRACTIONS)
    )
    # Keep the exact residual arithmetic used by production.  For example,
    # 0.3333 yields 0.59998 rather than silently snapping the conditioning
    # value to 0.6; metadata comparisons round only for human readability.
    train_progress = tuple(stage_progress(fractions))
    deployment_progress = tuple(stage_progress(deployment_fractions))
    ode_steps = int(_cfg_value("ode_steps", 4))
    deployment_ode_steps = int(_cfg_value("deployment_ode_steps", 4))
    modes = int(_cfg_value("geom_lowfreq_modes", 8))
    sigma_px = tuple(float(x) for x in _cfg_value("geom_sigma_px", EXPECTED_SIGMA_PX))
    sigma_feature = tuple(value / float(snake_config.down_ratio) for value in sigma_px)
    initial_noise_scale = float(_cfg_value("initial_noise_scale", 1.0))
    ppo_epochs = int(_cfg_value("ppo_inner_epochs", 2))
    ppo_clip = float(_cfg_value("ppo_clip", 0.05))
    ppo_kl_target = float(_cfg_value("ppo_kl_target", 0.002))
    kl_beta = float(_cfg_value("kl_beta", 0.01))
    adv_clip = float(_cfg_value("adv_clip_max", 2.0))
    adv_std_floor = float(_cfg_value("adv_std_floor", 0.1))
    grad_clip = float(_cfg_value("grad_clip_norm", 0.25))
    learning_rate = float(_cfg_value("lr", 4e-8))
    eval_panel_size = int(_cfg_value("eval_panel_size", 80))
    eval_every = int(_cfg_value("eval_every", 50))
    save_every = int(_cfg_value("save_every", 50))
    log_every = int(_cfg_value("log_every", 1))
    max_contours = int(_cfg_value("max_contours", 4))
    seed = int(_cfg_value("seed", 20260824))
    baseline_iou_floor = float(_cfg_value("min_baseline_iou", 0.45))
    baseline_dice_floor = float(_cfg_value("min_baseline_dice", 0.60))
    use_delta_nsd_reward = bool(_cfg_value("use_delta_nsd_reward", True))
    nsd_delta_px = float(_cfg_value("nsd_delta_px", EXPECTED_NSD_DELTA_PX))
    burr_weight = float(_cfg_value("reward_burr_weight", 0.06))
    burr_max_px = float(_cfg_value("reward_burr_max_px", 1.5))
    burr_margin_px = float(_cfg_value("reward_burr_margin_px", 0.5))
    burr_quantile = float(_cfg_value("reward_burr_quantile", 95.0))

    contract = {
        "backend": str(getattr(cfg, "detector_backend", "")),
        "replace": bool(getattr(cfg, "locate_feat_replace", False)),
        "inject": bool(getattr(cfg, "locate_feat_inject", False)),
        "feature_key": tuple(getattr(cfg, "locate_feat_keys", [])),
        "feature_dim": int(getattr(cfg, "locate_feat_dim", -1)),
        "fusion": str(getattr(cfg, "locate_feat_fusion_mode", "")),
        "fractions": tuple(round(x, 4) for x in fractions),
        "progress": tuple(round(x, 4) for x in train_progress),
        "deployment_fractions": tuple(round(x, 4) for x in deployment_fractions),
        "deployment_progress": tuple(round(x, 4) for x in deployment_progress),
        "ode_steps": ode_steps,
        "deployment_ode_steps": deployment_ode_steps,
        "solver": str(getattr(cfg, "flow_ode_solver", "")).lower(),
        "action_policy": str(_cfg_value("action_policy", "geom")),
        "modes": modes,
        "sigma_px": tuple(round(x, 4) for x in sigma_px),
        "reward_mode": "delta_nsd" if use_delta_nsd_reward else "invalid",
        "nsd_delta_px": round(nsd_delta_px, 4),
        "credit": str(_cfg_value("per_step_credit_mode", "full_extrap")),
        "credit_weight": float(_cfg_value("per_step_reward_weight", 1.0)),
        "group_centering": bool(_cfg_value("adv_center_group", True)),
        "flow_only": bool(_cfg_value("flow_only_update", True)),
    }
    expected_contract = {
        "backend": EXPECTED_BACKEND,
        "replace": True,
        "inject": False,
        "feature_key": ("layer_18",),
        "feature_dim": 1152,
        "fusion": "center_only",
        "fractions": EXPECTED_TRAIN_FRACTIONS,
        "progress": EXPECTED_TRAIN_PROGRESS,
        "deployment_fractions": EXPECTED_DEPLOYMENT_FRACTIONS,
        "deployment_progress": EXPECTED_DEPLOYMENT_PROGRESS,
        "ode_steps": 4,
        "deployment_ode_steps": 4,
        "solver": "ab2",
        "action_policy": "geom",
        "modes": 8,
        "sigma_px": EXPECTED_SIGMA_PX,
        "reward_mode": "delta_nsd",
        "nsd_delta_px": EXPECTED_NSD_DELTA_PX,
        "credit": "full_extrap",
        "credit_weight": 1.0,
        "group_centering": True,
        "flow_only": True,
    }
    _require_exact(contract, expected_contract, "RL scientific contract")
    _require_exact(int(_cfg_value("outer_steps", 5)), 5, "training outer stages")
    _require_exact(int(_cfg_value("deployment_outer_steps", 2)), 2, "deployment stages")
    _require_exact(k_rollouts, 8, "GRPO rollout count")
    _require_exact(ppo_epochs, 2, "PPO epochs")
    _require_exact(round(ppo_clip, 6), 0.05, "PPO clip")
    _require_exact(round(ppo_kl_target, 6), 0.002, "approx-KL stop")
    _require_exact(round(kl_beta, 6), 0.01, "explicit KL beta")
    _require_exact(round(adv_std_floor, 6), 0.1, "advantage std floor")
    _require_exact(round(grad_clip, 6), 0.25, "gradient clip")
    _require_exact(round(initial_noise_scale, 6), 1.0, "initial latent scale")
    _require_exact(int(cfg.train.batch_size), 1, "RL batch size")
    _require_exact(int(cfg.train.num_workers), 0, "RL loader workers")
    _require_exact(eval_panel_size, 80, "validation panel size")
    _require_exact(seed, 20260824, "training seed")
    if train_steps not in (2, 10_000):
        raise RuntimeError(f"train_steps must be preflight=2 or formal=10000, got {train_steps}")
    if eval_every != 50 or save_every != 50:
        raise RuntimeError("evaluation/checkpoint cadence drift")
    if not TUNE_MANIFEST.is_file() or sha256_file(TUNE_MANIFEST) != EXPECTED_TUNE_MANIFEST_SHA256:
        raise RuntimeError("immutable validation manifest is missing or changed")

    tune_catalog = DatasetCatalog.get("VolMemVal")
    tune_catalog["ann_file"] = str(TUNE_MANIFEST)
    DatasetCatalog.dataset_attrs[TUNE_DATASET] = tune_catalog
    _require_exact(str(cfg.test.dataset), TUNE_DATASET, "validation dataset")

    _set_seed(seed)
    source_path = _source_checkpoint()
    if len(EXPECTED_SOURCE_SHA256) != 64 or any(
        character not in "0123456789abcdef"
        for character in EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError(
            "launch through tools/launch_rl.py with an explicit source SHA256"
        )
    if EXPECTED_SOURCE_STEP <= 0:
        raise RuntimeError(
            "launch through tools/launch_rl.py with an explicit source step"
        )
    if sha256_file(source_path) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("stage-1 checkpoint SHA256 drift")
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    _require_exact(int(source_payload.get("step", -1)), EXPECTED_SOURCE_STEP, "source step")

    wrapper = _CheckpointWrapper(make_network(cfg)).cuda()
    source_state = _strict_load(wrapper, source_payload, "stage-1 checkpoint")
    del source_payload
    _require_exact(sum(p.numel() for p in wrapper.parameters()), EXPECTED_MODEL_PARAMETERS, "model parameters")
    inner = wrapper.net
    gcn = inner.gcn
    runtime = {
        "solver": getattr(gcn, "_ode_solver", None),
        "steps": int(getattr(gcn, "ode_steps", -1)),
        "noise": float(getattr(gcn, "_infer_noise_scale", float("nan"))),
        "samples": int(getattr(gcn, "_infer_avg_samples", -1)),
        "resample": bool(getattr(gcn, "_resample_feat_at_xt", False)),
        "latent_policy": bool(getattr(gcn, "_use_latent_policy", False)),
        "geom_bridge": bool(getattr(gcn, "_geom_bridge", False)),
        "inference_gate": bool(
            getattr(gcn, "_use_disp_gate", False)
            and getattr(gcn, "_disp_gate_apply_inference", False)
        ),
    }
    _require_exact(
        runtime,
        {
            "solver": "ab2",
            "steps": 4,
            "noise": 1.0,
            "samples": 1,
            "resample": False,
            "latent_policy": False,
            "geom_bridge": False,
            "inference_gate": False,
        },
        "runtime inference contract",
    )

    for parameter in wrapper.parameters():
        parameter.requires_grad = False
    for parameter in gcn.parameters():
        parameter.requires_grad = True
    inner.eval()
    gcn.train()
    frozen_bn = freeze_bn_running_stats(gcn)
    trainable = [parameter for parameter in gcn.parameters() if parameter.requires_grad]
    _require_exact(sum(p.numel() for p in trainable), EXPECTED_FLOW_PARAMETERS, "Flow trainable parameters")
    if inner.locate_feat_replacer is None:
        raise RuntimeError("MoonViT feature replacer is missing")
    _require_exact(
        sum(p.numel() for p in inner.locate_feat_replacer.parameters()),
        EXPECTED_CONTEXT_PARAMETERS,
        "frozen context parameters",
    )
    if any(p.requires_grad for p in inner.locate_feat_replacer.parameters()):
        raise RuntimeError("MoonViT replacer leaked into RL optimizer")

    # KL always references the signed supervised source, even when a run
    # resumes from a later RL checkpoint.
    reference_flow = copy.deepcopy(gcn).cuda().eval()
    for parameter in reference_flow.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=float(cfg.train.weight_decay)
    )
    start_step = 0
    resume_path_raw = os.environ.get("RL_RESUME_CHECKPOINT", "").strip()
    resume_path: Optional[Path] = None
    if resume_path_raw:
        resume_path = _project_path(resume_path_raw)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        _strict_load(wrapper, resume_payload, "RL resume checkpoint")
        if not isinstance(resume_payload.get("optimizer"), dict):
            raise RuntimeError("RL resume checkpoint has no optimizer state")
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload.get("step", -1))
        if not 0 < start_step < train_steps:
            raise RuntimeError(f"invalid RL resume step {start_step} for target {train_steps}")
        del resume_payload

    train_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    eval_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
    panel_size = min(eval_panel_size, len(eval_loader.dataset))
    panel_indices = np.linspace(0, len(eval_loader.dataset) - 1, panel_size, dtype=np.int64).tolist()
    panel_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(eval_loader.dataset, panel_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=eval_loader.collate_fn,
        pin_memory=True,
    )
    eval_batches = [_move_batch(batch) for batch in panel_loader]
    _require_exact(len(eval_batches), 80, "loaded validation panel")

    output_dir = _output_dir()
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / LOG_SUBDIR
    if checkpoint_dir.exists() or log_dir.exists():
        raise FileExistsError(f"refusing to reuse RL artifacts in {output_dir}")
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    log_path = log_dir / "logs.jsonl"
    eval_seed_base = 91_000_000 + seed
    expected_rms_px = [value * math.sqrt(modes / 128.0) for value in sigma_px]
    hparams = {
        "schema": "diffusionsnake.mainline.stage2_rl.v3",
        "cfg_file": _cfg_file_used(),
        "base_checkpoint": str(source_path),
        "base_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "resume_checkpoint": None if resume_path is None else str(resume_path),
        "seed": seed,
        "train_steps": train_steps,
        "k": k_rollouts,
        "outer_steps": len(fractions),
        "fractions": list(fractions),
        "stage_s_values": list(train_progress),
        "ode_steps": ode_steps,
        "training_schedule_role": "five_action_full_extrap_only",
        "deployment_schedule": {
            "outer_steps": len(deployment_fractions),
            "fractions": list(deployment_fractions),
            "stage_s_values": list(deployment_progress),
            "ode_steps_per_outer_stage": deployment_ode_steps,
            "solver": "ab2",
            "total_nfe": len(deployment_fractions) * deployment_ode_steps,
            "used_for_all_reported_evaluation": True,
        },
        "action_policy": "geom",
        "fourier_configuration": {
            "profile": "five_stage_m8_sigma_080_070_060_050_040",
            "modes": modes,
            "sigma_px": list(sigma_px),
            "expected_point_rms_px": expected_rms_px,
            "distribution": "standard_normal_low_frequency_coefficients",
            "direction": "contour_normal",
            "eval_dataset": TUNE_DATASET,
            "eval_panel_size": len(eval_batches),
            "eval_latent_policy": "fixed_per_panel_row_across_all_steps",
            "dev8_used_for_selection": False,
        },
        "credit_assignment": {
            "mode": "full_extrap",
            "outer_credit_map": [0, 1, 2, 3, 4],
            "stage_reward": "sampled_delta_nsd_minus_deterministic_delta_nsd_minus_sampled_burr",
            "advantage_centered_across_k": True,
            "advantage_std_floor": adv_std_floor,
        },
        "reward_mode": {
            "name": "delta_nsd",
            "delta_nsd": True,
            "nsd_delta_px": nsd_delta_px,
            "coordinate_space": "2d_image_pixels",
            "deterministic_reference": "same_stage_same_latent_flow",
            "composite_region_score_used_for_policy": False,
        },
        "burr": {
            "weight": burr_weight,
            "max_px": burr_max_px,
            "margin_px": burr_margin_px,
            "quantile": burr_quantile,
        },
        "ppo": {
            "epochs": ppo_epochs,
            "clip": ppo_clip,
            "approx_kl_stop": ppo_kl_target,
            "explicit_kl_beta": kl_beta,
            "gradient_clip": grad_clip,
            "learning_rate": learning_rate,
        },
        "parameters": {
            "total": EXPECTED_MODEL_PARAMETERS,
            "trainable_flow": EXPECTED_FLOW_PARAMETERS,
            "frozen_context": EXPECTED_CONTEXT_PARAMETERS,
        },
        "ha_smoe": {
            "routing_unit": "whole_contour",
            "routed_blocks": [2, 4, 6],
            "num_experts": 4,
            "top_k": 2,
            "shared_dense_ffn": True,
            "output_head": "dense",
        },
    }
    _atomic_json(log_dir / "hparams.json", hparams)

    def manual_context(batch: Dict) -> Dict:
        inner.eval()
        with torch.no_grad():
            if "locate_feat" not in batch:
                raise RuntimeError("MoonViT layer-18 cache is missing from batch")
            stride = max(int(round(inner.down_ratio)), 1)
            height = (int(batch["inp"].size(2)) + stride - 1) // stride
            width = (int(batch["inp"].size(3)) + stride - 1) // stride
            empty_feature = batch["inp"].new_zeros(batch["inp"].size(0), 1, height, width)
            cnn_feature, _ = inner.apply_locate_feature_replacement(empty_feature, batch)
        device = cnn_feature.device
        initial = _flatten_valid_polys(batch, "i_it_py", device)
        target = _flatten_valid_polys(batch, "i_gt_py", device)
        try:
            canonical = _flatten_valid_polys(batch, "c_it_py", device)
        except (KeyError, ValueError):
            canonical = snake_gcn_utils.img_poly_to_can_poly(initial)
        py_ind = _make_py_ind(batch, initial.size(0), device)
        count = min(initial.size(0), target.size(0), canonical.size(0), py_ind.size(0))
        initial, target, canonical, py_ind = (
            initial[:count], target[:count], canonical[:count], py_ind[:count]
        )
        if max_contours > 0:
            initial, target, canonical, py_ind = (
                initial[:max_contours], target[:max_contours], canonical[:max_contours], py_ind[:max_contours]
            )
        gcn.train()
        freeze_bn_running_stats(gcn)
        return {
            "cnn_feature": cnn_feature.detach(),
            "i_it_py": initial.detach(),
            "c_it_py": canonical.detach(),
            "i_gt_py": _align_gt(initial, target).detach(),
            "py_ind": py_ind.detach(),
            "image_hw": tuple(int(x) for x in batch["inp"].shape[-2:]),
        }

    def nsd_score(poly: torch.Tensor, target: torch.Tensor, image_hw) -> torch.Tensor:
        return compute_nsd_score(
            poly,
            target,
            H=image_hw[0],
            W=image_hw[1],
            delta_px=nsd_delta_px,
            coord_scale=float(snake_config.down_ratio),
        )

    def metric(poly: torch.Tensor, target: torch.Tensor, image_hw, name: str) -> torch.Tensor:
        if name == "nsd":
            return nsd_score(poly, target, image_hw)
        weights = {
            "iou": (0.0, 0.0, 1.0, 0.0),
            "dice": (0.0, 1.0, 0.0, 0.0),
            "mboundf": (1.0, 0.0, 0.0, 0.0),
        }[name]
        return compute_region_score(
            poly,
            target,
            H=image_hw[0],
            W=image_hw[1],
            w_boundary=weights[0],
            w_dice=weights[1],
            w_iou=weights[2],
            w_dist=weights[3],
            coord_scale=float(snake_config.down_ratio),
        )

    def new_latents(output: Dict, count: int) -> list[torch.Tensor]:
        return [torch.randn_like(output["i_it_py"]) * initial_noise_scale for _ in range(count)]

    def deterministic_rollout(
        output: Dict,
        flow: nn.Module,
        schedule: Sequence[float],
        progress_values: Sequence[float],
        steps_per_stage: int,
        latents: Sequence[torch.Tensor],
    ) -> Dict:
        if len(schedule) != len(latents):
            raise RuntimeError("latent count does not match outer-stage count")
        current = output["i_it_py"].detach()
        total_disp = torch.zeros_like(current)
        polygons = [current]
        for index, fraction in enumerate(schedule):
            canonical = snake_gcn_utils.img_poly_to_can_poly(current)
            action = outer_action_mean(
                flow,
                output["cnn_feature"],
                current,
                canonical,
                output["py_ind"],
                fraction,
                steps_per_stage,
                progress_values[index],
                latents[index],
            )
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
            polygons.append(current)
        return {
            "py": output["i_it_py"] + total_disp,
            "disp": total_disp,
            "polys": polygons,
        }

    @torch.no_grad()
    def sample_rollout(output: Dict, shared_latents: Sequence[torch.Tensor]) -> Dict:
        current = output["i_it_py"].detach()
        trajectory = {
            "states": [],
            "c_states": [],
            "actions": [],
            "old_logs": [],
            "outer_latents": [],
            "polys": [current],
        }
        for index, (fraction, sigma) in enumerate(zip(fractions, sigma_feature)):
            canonical = snake_gcn_utils.img_poly_to_can_poly(current)
            mean = outer_action_mean(
                gcn,
                output["cnn_feature"],
                current,
                canonical,
                output["py_ind"],
                fraction,
                ode_steps,
                train_progress[index],
                shared_latents[index],
            )
            coefficients = torch.randn(current.size(0), modes, device=current.device, dtype=current.dtype)
            action = mean + low_frequency_delta(current, coefficients, sigma)
            trajectory["states"].append(current)
            trajectory["c_states"].append(canonical.detach())
            trajectory["actions"].append(action.detach())
            trajectory["old_logs"].append(standard_normal_logprob(coefficients).detach())
            trajectory["outer_latents"].append(shared_latents[index].detach())
            current = (current + action).detach()
            trajectory["polys"].append(current)
        if len(trajectory["actions"]) != 5:
            raise RuntimeError("each rollout must contain exactly five RL actions")
        trajectory["py"] = current
        trajectory["disp"] = current - output["i_it_py"]
        return trajectory

    @torch.no_grad()
    def compute_eval() -> dict:
        values = {name: [] for name in ("iou", "dice", "mboundf", "nsd")}
        for index, batch in enumerate(eval_batches):
            output = manual_context(batch)
            if output["i_it_py"].numel() == 0:
                continue
            device_index = output["i_it_py"].device.index
            devices = [] if device_index is None else [int(device_index)]
            with torch.random.fork_rng(devices=devices):
                eval_seed = eval_seed_base + index
                torch.manual_seed(eval_seed)
                torch.cuda.manual_seed(eval_seed)
                latents = new_latents(output, len(deployment_fractions))
            prediction = deterministic_rollout(
                output, gcn, deployment_fractions, deployment_progress, deployment_ode_steps, latents
            )["py"]
            for name in values:
                values[name].append(metric(prediction, output["i_gt_py"], output["image_hw"], name))
        if not values["iou"]:
            return {}
        merged = {name: torch.cat(parts) for name, parts in values.items()}
        return {
            "eval_iou": _finite_scalar(merged["iou"].mean(), "eval_iou"),
            "eval_dice": _finite_scalar(merged["dice"].mean(), "eval_dice"),
            "eval_mboundf": _finite_scalar(merged["mboundf"].mean(), "eval_mboundf"),
            "eval_nsd_2px": _finite_scalar(merged["nsd"].mean(), "eval_nsd_2px"),
            "eval_n": int(merged["iou"].numel()),
        }

    def audit_frozen_state() -> None:
        current = wrapper.state_dict()
        changed = [
            name
            for name, tensor in source_state.items()
            if not name.startswith("net.gcn.")
            and not torch.equal(tensor.cpu(), current[name].detach().cpu())
        ]
        if changed:
            raise RuntimeError(f"frozen tensors changed: {changed[:8]}")

    def save_checkpoint(path: Path, step: int, metrics: dict) -> None:
        audit_frozen_state()
        state = wrapper.state_dict()
        nonfinite = [
            name
            for name, tensor in state.items()
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item()
        ]
        if nonfinite:
            raise FloatingPointError(f"checkpoint has non-finite tensors: {nonfinite[:8]}")
        _atomic_torch_save(
            {
                "format_version": 1,
                "state_dict": state,
                "optimizer": optimizer.state_dict(),
                "step": int(step),
                "metrics": metrics,
                "cfg_file": _cfg_file_used(),
                "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
                "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiment_metadata": {
                    "pipeline": "five_action_fourier_grpo_2x4_deploy",
                    "train_fractions": list(fractions),
                    "train_progress": list(train_progress),
                    "deployment_fractions": list(deployment_fractions),
                    "deployment_progress": list(deployment_progress),
                    "fourier_modes": modes,
                    "fourier_sigma_px": list(sigma_px),
                    "reward_mode": "delta_nsd",
                    "nsd_delta_px": nsd_delta_px,
                    "flow_only": True,
                    "frozen_bn_layers": frozen_bn,
                },
            },
            path,
        )

    baseline_eval = compute_eval()
    if not baseline_eval:
        raise RuntimeError("fixed validation panel produced no contours")
    if baseline_eval["eval_iou"] < baseline_iou_floor or baseline_eval["eval_dice"] < baseline_dice_floor:
        raise RuntimeError(f"supervised baseline sanity gate failed: {baseline_eval}")
    _atomic_json(
        log_dir / "eval_baseline_step0.json",
        {
            "step": 0,
            "training_seed": seed,
            "eval_seed_base": eval_seed_base,
            "eval_latent_policy": "fixed_per_panel_row_across_all_steps",
            **baseline_eval,
        },
    )
    print(
        "[RL] baseline panel: IoU={eval_iou:.6f} Dice={eval_dice:.6f} "
        "mBoundF={eval_mboundf:.6f} NSD@2px={eval_nsd_2px:.6f} "
        "n={eval_n}".format(**baseline_eval),
        flush=True,
    )

    train_iterator = iter(train_loader)
    alignment_checked = False
    reward_ema = EMA(decay=0.95)
    best_eval_iou = baseline_eval["eval_iou"]
    step = start_step
    empty_streak = 0
    max_empty_streak = max(len(train_loader), 32)
    while step < train_steps:
        freeze_bn_running_stats(gcn)
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        output = manual_context(_move_batch(batch))
        if output["i_it_py"].numel() == 0 or output["i_gt_py"].numel() == 0:
            empty_streak += 1
            if empty_streak > max_empty_streak:
                raise RuntimeError("too many consecutive empty-contour batches")
            continue
        empty_streak = 0
        step += 1

        if not alignment_checked:
            with torch.no_grad():
                cuda_state = torch.cuda.get_rng_state()
                model_training_disp = gcn.sample_disp_iterative(
                    output["cnn_feature"],
                    output["i_it_py"],
                    snake_gcn_utils.img_poly_to_can_poly(output["i_it_py"]),
                    output["py_ind"],
                    num_iter_steps=len(fractions),
                    fractions=fractions,
                    ode_steps=ode_steps,
                )
                torch.cuda.set_rng_state(cuda_state)
                train_latents = new_latents(output, len(fractions))
                helper_training_disp = deterministic_rollout(
                    output, gcn, fractions, train_progress, ode_steps, train_latents
                )["disp"]
                training_error = (model_training_disp - helper_training_disp).abs().max().item()

                cuda_state = torch.cuda.get_rng_state()
                model_deployment_disp = gcn.sample_disp_iterative(
                    output["cnn_feature"],
                    output["i_it_py"],
                    snake_gcn_utils.img_poly_to_can_poly(output["i_it_py"]),
                    output["py_ind"],
                    num_iter_steps=len(deployment_fractions),
                    fractions=deployment_fractions,
                    ode_steps=deployment_ode_steps,
                )
                torch.cuda.set_rng_state(cuda_state)
                deploy_latents = new_latents(output, len(deployment_fractions))
                helper_deployment_disp = deterministic_rollout(
                    output,
                    gcn,
                    deployment_fractions,
                    deployment_progress,
                    deployment_ode_steps,
                    deploy_latents,
                )["disp"]
                deployment_error = (model_deployment_disp - helper_deployment_disp).abs().max().item()
            if not math.isfinite(training_error) or training_error > 1e-5:
                raise RuntimeError(f"five-stage AB2 training rollout alignment failed: {training_error}")
            if not math.isfinite(deployment_error) or deployment_error > 1e-5:
                raise RuntimeError(f"production 2x4 AB2 deployment alignment failed: {deployment_error}")
            print(
                f"[RL] five-stage AB2 training rollout alignment PASS max_abs={training_error:.9g} "
                f"stage_s={list(train_progress)}",
                flush=True,
            )
            print(
                f"[RL] production 2x4 AB2 deployment alignment PASS max_abs={deployment_error:.9g} "
                f"stage_s={list(deployment_progress)}",
                flush=True,
            )
            alignment_checked = True

        shared_latents = new_latents(output, len(fractions))
        deterministic = deterministic_rollout(
            output, gcn, fractions, train_progress, ode_steps, shared_latents
        )
        baseline_nsd = nsd_score(
            deterministic["py"], output["i_gt_py"], output["image_hw"]
        ).detach()
        rollouts = [sample_rollout(output, shared_latents) for _ in range(k_rollouts)]
        final_nsd_scores, rewards, burr_values = [], [], []
        for rollout in rollouts:
            final_nsd = nsd_score(
                rollout["py"], output["i_gt_py"], output["image_hw"]
            ).detach()
            burr, _ = _burr_penalty(
                rollout["py"],
                output["i_gt_py"],
                float(snake_config.down_ratio),
                burr_margin_px,
                burr_max_px,
                burr_quantile,
            )
            final_nsd_scores.append(final_nsd)
            burr_values.append(burr.detach())
            rewards.append(
                compute_delta_nsd_reward(
                    final_nsd,
                    baseline_nsd,
                    burr.detach(),
                    burr_weight=burr_weight,
                )
            )
        quality = torch.stack(rewards)
        quality_std = quality.std(dim=0, unbiased=False, keepdim=True)
        terminal_advantage = (
            (quality - quality.mean(dim=0, keepdim=True))
            / quality_std.clamp_min(adv_std_floor)
        ).clamp(-adv_clip, adv_clip)

        # Full-extrap: judge every outer action with the same delta-NSD@2px
        # contract as the terminal trajectory.  The sampled and deterministic
        # endpoints share the stage state and latent; only the Fourier action
        # differs. The sampled endpoint also keeps the released burr penalty.
        with torch.no_grad():
            deterministic_step_nsd = []
            for index, fraction in enumerate(fractions):
                start, end = deterministic["polys"][index : index + 2]
                deterministic_step_nsd.append(
                    nsd_score(
                        start + (end - start) / fraction,
                        output["i_gt_py"],
                        output["image_hw"],
                    )
                )
            deterministic_step_nsd = torch.stack(deterministic_step_nsd)
            rollout_step_rewards = []
            rollout_step_burrs = []
            for rollout in rollouts:
                rewards_at_stage = []
                burrs_at_stage = []
                for index, fraction in enumerate(fractions):
                    start, end = rollout["polys"][index : index + 2]
                    extrapolated = start + (end - start) / fraction
                    extrapolated_nsd = nsd_score(
                        extrapolated,
                        output["i_gt_py"],
                        output["image_hw"],
                    )
                    extrapolated_burr, _ = _burr_penalty(
                        extrapolated,
                        output["i_gt_py"],
                        float(snake_config.down_ratio),
                        burr_margin_px,
                        burr_max_px,
                        burr_quantile,
                    )
                    rewards_at_stage.append(
                        compute_delta_nsd_reward(
                            extrapolated_nsd,
                            deterministic_step_nsd[index],
                            extrapolated_burr.detach(),
                            burr_weight=burr_weight,
                        )
                    )
                    burrs_at_stage.append(extrapolated_burr.detach())
                rollout_step_rewards.append(torch.stack(rewards_at_stage))
                rollout_step_burrs.append(torch.stack(burrs_at_stage))
            step_quality = torch.stack(rollout_step_rewards)
            step_burr = torch.stack(rollout_step_burrs)
            step_std = step_quality.std(dim=0, unbiased=False, keepdim=True)
            step_advantage = (
                (step_quality - step_quality.mean(dim=0, keepdim=True))
                / step_std.clamp_min(adv_std_floor)
            ).clamp(-adv_clip, adv_clip)

        approx_kl_history, loss_history, ratio_history = [], [], []
        early_stop_epoch = ppo_epochs
        total_actions = k_rollouts * len(fractions)
        grad_norm = torch.zeros((), device="cuda")
        for epoch in range(ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            epoch_losses, epoch_kls, epoch_ratios = [], [], []
            for rollout_index, rollout in enumerate(rollouts):
                for action_index, action in enumerate(rollout["actions"]):
                    state = rollout["states"][action_index]
                    canonical = rollout["c_states"][action_index]
                    mean = outer_action_mean(
                        gcn,
                        output["cnn_feature"],
                        state,
                        canonical,
                        output["py_ind"],
                        fractions[action_index],
                        ode_steps,
                        train_progress[action_index],
                        rollout["outer_latents"][action_index],
                    )
                    logprob = fourier_action_logprob(
                        action, mean, state, sigma_feature[action_index], modes
                    )
                    old_logprob = rollout["old_logs"][action_index]
                    ratio = torch.exp(logprob - old_logprob)
                    advantage = step_advantage[rollout_index, action_index].detach()
                    unclipped = -advantage * ratio
                    clipped = -advantage * ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
                    policy_loss = torch.maximum(unclipped, clipped).mean() / total_actions
                    with torch.no_grad():
                        reference_mean = outer_action_mean(
                            reference_flow,
                            output["cnn_feature"],
                            state,
                            canonical,
                            output["py_ind"],
                            fractions[action_index],
                            ode_steps,
                            train_progress[action_index],
                            rollout["outer_latents"][action_index],
                        )
                    kl_loss = fourier_mean_kl(
                        mean, reference_mean, state, sigma_feature[action_index], modes
                    ).mean() / total_actions
                    loss = policy_loss + kl_beta * kl_loss
                    if not torch.isfinite(loss).all().item():
                        raise FloatingPointError(f"non-finite PPO loss at step {step}")
                    loss.backward()
                    log_delta = logprob.detach() - old_logprob
                    epoch_losses.append(policy_loss.detach())
                    epoch_kls.append(0.5 * log_delta.square().mean())
                    epoch_ratios.append(ratio.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            mean_kl = torch.stack(epoch_kls).mean()
            approx_kl_history.append(mean_kl)
            loss_history.extend(epoch_losses)
            ratio_history.extend(epoch_ratios)
            if _finite_scalar(mean_kl, "approx_kl") > ppo_kl_target:
                early_stop_epoch = epoch + 1
                break

        ratio_tensor = torch.cat(ratio_history)
        metrics = {
            "step": step,
            "reward_mean": _finite_scalar(quality.mean(), "reward_mean"),
            "reward_std_mean": _finite_scalar(quality_std.mean(), "reward_std_mean"),
            "reward_ema": reward_ema.update(float(quality.mean().item())),
            "terminal_advantage_abs_mean": _finite_scalar(
                terminal_advantage.abs().mean(), "terminal_advantage_abs_mean"
            ),
            "step_quality_std_mean": _finite_scalar(step_std.mean(), "step_quality_std_mean"),
            "step_advantage_abs_mean": _finite_scalar(
                step_advantage.abs().mean(), "step_advantage_abs_mean"
            ),
            "policy_loss": _finite_scalar(torch.stack(loss_history).mean(), "policy_loss"),
            "grad_norm": _finite_scalar(grad_norm, "grad_norm"),
            "approx_kl": _finite_scalar(torch.stack(approx_kl_history).mean(), "approx_kl"),
            "ratio_mean": _finite_scalar(ratio_tensor.mean(), "ratio_mean"),
            "ratio_min": _finite_scalar(ratio_tensor.min(), "ratio_min"),
            "ratio_max": _finite_scalar(ratio_tensor.max(), "ratio_max"),
            "early_stop_epoch": early_stop_epoch,
            "reward_mode": "delta_nsd@2px",
            "baseline_nsd_mean": _finite_scalar(baseline_nsd.mean(), "baseline_nsd_mean"),
            "final_nsd_mean": _finite_scalar(
                torch.stack(final_nsd_scores).mean(), "final_nsd_mean"
            ),
            "burr_penalty_mean": _finite_scalar(torch.stack(burr_values).mean(), "burr_penalty_mean"),
            "step_burr_penalty_mean": _finite_scalar(
                step_burr.mean(), "step_burr_penalty_mean"
            ),
            "quality_best_mean": _finite_scalar(quality.max(dim=0).values.mean(), "quality_best_mean"),
            "quality_p10": percentiles(quality.flatten())["p10"],
            "quality_p50": percentiles(quality.flatten())["p50"],
            "quality_p90": percentiles(quality.flatten())["p90"],
            "outer_log_count_mean": 5.0,
            "action_policy": "geom",
            "geom_lowfreq_modes": modes,
            "geom_sigma_px": list(sigma_px),
            "fourier_profile": hparams["fourier_configuration"]["profile"],
            "learning_rate": learning_rate,
        }
        eval_metrics = {}
        if step % eval_every == 0:
            eval_metrics = compute_eval()
            metrics.update(eval_metrics)
            if eval_metrics["eval_iou"] > best_eval_iou:
                best_eval_iou = eval_metrics["eval_iou"]
                save_checkpoint(checkpoint_dir / "best_iou.pt", step, metrics)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        if step % log_every == 0:
            print(
                f"[RL] step={step}/{train_steps} reward={metrics['reward_mean']:+.6f} "
                f"std={metrics['reward_std_mean']:.6f} loss={metrics['policy_loss']:+.6f} "
                f"grad={metrics['grad_norm']:.6f} kl={metrics['approx_kl']:.6f}",
                flush=True,
            )
        if step == 1 or step % save_every == 0:
            save_checkpoint(checkpoint_dir / "latest.pt", step, metrics)
            if step % save_every == 0:
                save_checkpoint(checkpoint_dir / f"step_{step}.pt", step, metrics)

        del batch, output, deterministic, rollouts, quality, step_quality, step_burr
        gc.collect()
        torch.cuda.empty_cache()

    final_eval = compute_eval()
    save_checkpoint(
        checkpoint_dir / "latest.pt",
        train_steps,
        {"step": train_steps, "final_eval": final_eval},
    )
    print(f"[RL] completed: {output_dir}")


if __name__ == "__main__":
    main()
