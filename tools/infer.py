#!/usr/bin/env python3
"""Direct detector-free inference for the physical Memory-free Pure2D mainline.

This entry point deliberately builds only the signed Pure2D diffusion-snake
network.  It consumes GT boxes/classes for scientific evaluation, converts
each rectangle to the Route-B 12-point octagon, and runs the fixed AB2 8-NFE
trajectory.  No detector and no Memory wrapper are constructed.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import sys
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.runtime import require_idle_gpu


DEV8_CASES = (
    "sub-verse004",
    "sub-verse005",
    "sub-verse033",
    "sub-verse043",
    "sub-verse060",
    "sub-verse064",
    "sub-verse082",
    "sub-verse410_split-verse267",
)
LOCKED_CASES = {"sub-verse010", "sub-verse011", "sub-verse013"}
EXPECTED_SLICES = 1123
EXPECTED_PARAMETERS = 16933968


def parse_args(argv=None):
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--config", default=str(project_root / "configs/stage1.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--metric-module",
        default=str(project_root / "lib/evaluation/verse3d.py"),
    )
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--case-metadata", required=True)
    parser.add_argument("--locate-feat-cache-root", required=True)
    parser.add_argument(
        "--expected-parameters",
        type=int,
        default=EXPECTED_PARAMETERS,
        help="fail-closed total model parameter gate",
    )
    parser.add_argument(
        "--expected-detector-backend",
        default="flow_box_only",
        choices=("flow_box_only",),
        help="fail-closed detector-free feature backend gate",
    )
    parser.add_argument(
        "--teams-metric-module",
        default=str(project_root / "lib/evaluation/teams.py"),
        help="TEAMS-style 2D metric adapter",
    )
    parser.add_argument(
        "--teams-upstream-root",
        help="pinned local TEAMS checkout used to record upstream metric provenance",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="physical GPU; it must pass two strict idle samples 15 seconds apart",
    )
    parser.add_argument(
        "--max-idle-memory-mib",
        type=int,
        default=20,
        help=(
            "maximum driver-resident memory allowed by the two-sample GPU "
            "idle gate; the resolved threshold is recorded in result provenance"
        ),
    )
    parser.add_argument(
        "--smoke-slices",
        type=int,
        default=0,
        help=(
            "run the first N foreground Dev8 rows as a non-reportable smoke test; "
            "0 evaluates the complete 1123-row cohort"
        ),
    )
    return parser.parse_args(argv)


def foreground_smoke_indices(records, count):
    """Select foreground rows in manifest order for an end-to-end smoke test."""

    selected = []
    for index, record in enumerate(records):
        mask_path = os.fspath(record["mask_path"])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError("smoke mask missing: {}".format(mask_path))
        if np.any(mask > 0):
            selected.append(index)
            if len(selected) == int(count):
                return selected
    raise RuntimeError(
        "requested {} foreground smoke slices, found only {}".format(
            int(count), len(selected)
        )
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load {} from {}".format(name, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def move_batch(batch, device):
    for key, value in list(batch.items()):
        if key == "locate_feat" or str(key).startswith("locate_feat_"):
            continue
        if torch.is_tensor(value):
            batch[key] = value.to(device=device, non_blocking=True)
    if "locate_feat" in batch:
        batch["locate_feat"] = [
            feature.to(device=device, dtype=torch.float16, non_blocking=True)
            for feature in batch["locate_feat"]
        ]
    return batch


def strict_load(wrapper, checkpoint_path):
    from lib.checkpoints import extract_state_dict, normalize_state_dict

    checkpoint = torch.load(os.fspath(checkpoint_path), map_location="cpu")
    normalized = normalize_state_dict(extract_state_dict(checkpoint))
    target = wrapper.state_dict()
    missing = sorted(set(target).difference(normalized))
    unexpected = sorted(set(normalized).difference(target))
    shape_mismatch = sorted(
        key for key in set(target).intersection(normalized)
        if tuple(target[key].shape) != tuple(normalized[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "strict checkpoint mismatch missing={} unexpected={} shape={}".format(
                missing, unexpected, shape_mismatch
            )
        )
    wrapper.load_state_dict(normalized, strict=True)
    return checkpoint, len(normalized)


def make_evaluator(Evaluator, cfg, dataset, result_dir, path_key):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = cfg
    evaluator.result_dir = os.path.abspath(os.fspath(result_dir))
    os.makedirs(evaluator.result_dir, exist_ok=True)
    evaluator.results = []
    evaluator._class_stats = {}
    evaluator.data_root = os.path.abspath(os.fspath(dataset.data_root))
    evaluator.ann_file = os.path.abspath(os.fspath(dataset.ann_file))
    evaluator.catalog_split = "dev"
    evaluator.box_mode = "gt"
    evaluator._records_by_image = {}
    for row in dataset.records:
        record = {
            "case_id": str(row["case_id"]),
            "slice_idx": int(row["slice_idx"]),
            "image_path": os.path.abspath(os.fspath(row["image_path"])),
            "mask_path": os.path.abspath(os.fspath(row["mask_path"])),
            "row_number": int(row.get("row_number", -1)),
        }
        key = path_key(record["image_path"], evaluator.data_root)
        if key in evaluator._records_by_image:
            raise ValueError("duplicate image path {}".format(record["image_path"]))
        evaluator._records_by_image[key] = record
    return evaluator


def prediction_masks(output, batch, evaluator, inverse_affine_points, snake_config):
    batch_size = int(batch["inp"].size(0))
    predictions = evaluator._prepare_predictions(output, batch_size)
    image_paths = batch["img_path"]
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    masks = []
    for sample_index, sample_predictions in enumerate(predictions):
        record = evaluator._record_for_path(image_paths[sample_index])
        gt_mask = evaluator._read_mask(record["mask_path"])
        _, inv_trans, orig_hw, flipped = evaluator._sample_metadata(
            batch, sample_index, batch_size, record, gt_mask.shape
        )
        label_mask = np.zeros(gt_mask.shape, dtype=np.uint16)
        for contour, label, _ in sample_predictions:
            restored = inverse_affine_points(
                contour * float(snake_config.down_ratio),
                inv_trans,
                orig_hw,
                flipped=flipped,
            )
            polygon = np.rint(restored).astype(np.int32)
            if polygon.shape[0] >= 3:
                cv2.fillPoly(label_mask, [polygon], int(label) + 1)
        masks.append(label_mask)
    return masks


def label_colors():
    colors = np.zeros((256, 3), dtype=np.uint8)
    for label in range(1, 256):
        hsv = np.uint8([[[int((label * 37) % 180), 210, 245]]])
        colors[label] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return colors


def gray_bgr(image):
    image = np.asarray(image)
    finite = image[np.isfinite(image)]
    if finite.size:
        lo, hi = np.percentile(finite, [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def overlay_labels(base, labels, colors, alpha=0.46):
    result = base.copy()
    foreground = labels > 0
    if foreground.any():
        color = colors[np.asarray(labels, dtype=np.uint8)]
        result[foreground] = np.rint(
            (1.0 - alpha) * result[foreground].astype(np.float32)
            + alpha * color[foreground].astype(np.float32)
        ).astype(np.uint8)
    return result


def title_panel(panel, title):
    panel = cv2.copyMakeBorder(panel, 42, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    return panel


def draw_route_b_init(panel, gt, get_quadrangle, get_octagon):
    for label in [int(x) for x in np.unique(gt) if int(x) > 0]:
        yy, xx = np.where(gt == label)
        if not len(xx):
            continue
        box = np.asarray([xx.min(), yy.min(), xx.max(), yy.max()], dtype=np.float32)
        octagon = get_octagon(get_quadrangle(box))
        cv2.rectangle(
            panel,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 180, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.polylines(
            panel,
            [np.rint(octagon).astype(np.int32)],
            True,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return panel


def make_visualization(path, evaluator, collector, get_quadrangle, get_octagon):
    candidates = sorted(
        [row for row in evaluator.results if int(row["gt_foreground_pixels"]) > 0],
        key=lambda row: float(row["foreground_dice"]),
    )
    if not candidates:
        raise RuntimeError("no foreground slices available for visualization")
    indices = [max(0, int(round(0.10 * (len(candidates) - 1)))), int(round(0.50 * (len(candidates) - 1)))]
    chosen = [candidates[index] for index in indices]
    colors = label_colors()
    rows = []
    selected = []
    for rank, result in zip(("hard-P10", "typical-P50"), chosen):
        case_id = str(result["case_id"])
        slice_idx = int(result["slice_idx"])
        record = evaluator._record_for_path(result["image_path"])
        image = cv2.imread(record["image_path"], cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(record["mask_path"], cv2.IMREAD_UNCHANGED)
        if image is None or gt is None:
            raise FileNotFoundError("visualization input is missing")
        pred = collector.canonical[case_id][slice_idx].T.astype(np.uint16, copy=False)
        base = gray_bgr(image)
        init_panel = draw_route_b_init(base.copy(), gt, get_quadrangle, get_octagon)
        gt_panel = overlay_labels(base, gt, colors)
        pred_panel = overlay_labels(base, pred, colors)
        error = base.copy()
        gt_fg = gt > 0
        pred_fg = pred > 0
        inter = gt_fg & pred_fg
        false_pos = (~gt_fg) & pred_fg
        false_neg = gt_fg & (~pred_fg)
        error[inter] = np.rint(0.35 * error[inter] + 0.65 * np.asarray([0, 210, 0])).astype(np.uint8)
        error[false_pos] = np.asarray([20, 20, 245], dtype=np.uint8)
        error[false_neg] = np.asarray([245, 80, 20], dtype=np.uint8)
        panels = [init_panel, gt_panel, pred_panel, error]
        labels = [
            "Input + Route-B init",
            "Ground truth",
            "Prediction",
            "Error: TP green / FP red / FN blue",
        ]
        panels = [cv2.resize(item, (430, 430), interpolation=cv2.INTER_AREA) for item in panels]
        panels = [title_panel(item, title) for item, title in zip(panels, labels)]
        row_canvas = np.concatenate(panels, axis=1)
        footer = np.full((45, row_canvas.shape[1], 3), 18, dtype=np.uint8)
        caption = "{}  {} x{:04d}  slice Dice={:.4f}  IoU={:.4f}".format(
            rank, case_id, slice_idx, float(result["foreground_dice"]), float(result["foreground_iou"])
        )
        cv2.putText(footer, caption, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
        rows.append(np.concatenate([row_canvas, footer], axis=0))
        selected.append({
            "rank": rank,
            "case_id": case_id,
            "slice_idx": slice_idx,
            "foreground_dice": float(result["foreground_dice"]),
            "foreground_iou": float(result["foreground_iou"]),
            "image_path": record["image_path"],
            "mask_path": record["mask_path"],
        })
    canvas = np.concatenate(rows, axis=0)
    if not cv2.imwrite(os.fspath(path), canvas):
        raise RuntimeError("failed to write visualization {}".format(path))
    return selected


def main(argv=None):
    args = parse_args(argv)
    project_root = pathlib.Path(args.project_root).resolve()
    config_path = pathlib.Path(args.config).resolve()
    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    result_dir = pathlib.Path(args.result_dir).resolve()
    metric_path = pathlib.Path(args.metric_module).resolve()
    slice_manifest = pathlib.Path(args.slice_manifest).resolve()
    data_root = pathlib.Path(args.data_root).resolve()
    case_metadata = pathlib.Path(args.case_metadata).resolve()
    locate_root = pathlib.Path(args.locate_feat_cache_root).resolve()
    teams_metric_path = (
        pathlib.Path(args.teams_metric_module).resolve()
        if args.teams_metric_module else None
    )
    teams_upstream_root = (
        pathlib.Path(args.teams_upstream_root).resolve()
        if args.teams_upstream_root else None
    )
    for path, label in (
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
        (metric_path, "metric module"),
        (slice_manifest, "slice manifest"),
        (case_metadata, "case metadata"),
    ):
        if not path.is_file():
            raise FileNotFoundError("{} missing: {}".format(label, path))
    if not locate_root.is_dir():
        raise FileNotFoundError("MoonViT cache missing: {}".format(locate_root))
    if not data_root.is_dir():
        raise FileNotFoundError("dataset root missing: {}".format(data_root))
    if teams_metric_path is not None and not teams_metric_path.is_file():
        raise FileNotFoundError("TEAMS metric module missing: {}".format(teams_metric_path))
    if teams_upstream_root is not None and not teams_upstream_root.is_dir():
        raise FileNotFoundError("TEAMS upstream root missing: {}".format(teams_upstream_root))
    if result_dir.exists():
        raise FileExistsError("result directory must be fresh: {}".format(result_dir))
    gpu_checks = require_idle_gpu(
        args.gpu,
        max_memory_mib=int(args.max_idle_memory_mib),
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    result_dir.mkdir(parents=True)

    os.chdir(os.fspath(project_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ["CFG_FILE"] = os.fspath(config_path)
    os.environ["DIFFUSIONSNAKE_DATA_ROOT"] = os.fspath(data_root)
    os.environ["DIFFUSIONSNAKE_SLICE_MANIFEST"] = os.fspath(slice_manifest)
    sys.argv = [sys.argv[0], "--cfg_file", os.fspath(config_path)]

    from lib.config import cfg
    from lib.datasets.collate_batch import snake_collator
    from lib.datasets.dataset_catalog import DatasetCatalog
    from lib.datasets.sagittal_2d_fixed.snake import Dataset
    from lib.evaluators.sagittal_2d_fixed.snake import (
        Evaluator,
        _path_key,
        inverse_affine_points,
    )
    from lib.networks import make_network
    from lib.train.trainers.make_trainer import _wrapper_factory
    from lib.utils.snake import snake_config
    from lib.utils.snake.snake_voc_utils import get_octagon, get_quadrangle

    cfg.use_grpo = False
    cfg.use_gt_det = True
    cfg.use_gt_det_train_only = False
    cfg.sagittal_eval_box_mode = "gt"
    configured_backend = str(getattr(cfg, "detector_backend", ""))
    if configured_backend != str(args.expected_detector_backend):
        raise RuntimeError(
            "detector backend gate failed: config={} expected={}".format(
                configured_backend, args.expected_detector_backend
            )
        )
    cfg.detector_backend = str(args.expected_detector_backend)
    cfg.locate_feat_cache_root = os.fspath(locate_root)
    cfg.sagittal_moonvit_cache_root = os.fspath(locate_root)
    cfg.image_color_aug = False
    cfg.image_lr_flip = False
    cfg.image_random_crop = False
    cfg.flow_ode_steps = 4
    cfg.iterative_num_steps = 2
    cfg.iterative_fractions = [0.6667, 1.0]
    cfg.iterative_ode_steps = 4
    cfg.flow_ode_solver = "ab2"

    attrs = DatasetCatalog.get("VolMemDev8")
    attrs.pop("id", None)
    dataset = Dataset(**attrs)
    cases = sorted({str(row["case_id"]) for row in dataset.records})
    if len(dataset) != EXPECTED_SLICES or tuple(cases) != tuple(sorted(DEV8_CASES)):
        raise RuntimeError("Dev8 identity drift: slices={} cases={}".format(len(dataset), cases))
    if LOCKED_CASES.intersection(cases):
        raise RuntimeError("locked cases selected")

    smoke_slices = int(args.smoke_slices)
    if smoke_slices < 0 or smoke_slices > EXPECTED_SLICES:
        raise ValueError(
            "--smoke-slices must be between 0 and {}".format(EXPECTED_SLICES)
        )
    smoke_mode = smoke_slices > 0
    expected_processed = smoke_slices if smoke_mode else EXPECTED_SLICES
    smoke_indices = (
        foreground_smoke_indices(dataset.records, smoke_slices)
        if smoke_mode else []
    )
    loader_dataset = (
        torch.utils.data.Subset(dataset, smoke_indices)
        if smoke_mode else dataset
    )

    loader = torch.utils.data.DataLoader(
        loader_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=snake_collator,
        pin_memory=True,
        persistent_workers=bool(args.num_workers > 0),
    )

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this inference requires an available CUDA device")
    device = torch.device("cuda")
    seed_all(args.seed)
    base_network = make_network(cfg)
    wrapper = _wrapper_factory(cfg, base_network)
    parameter_count = int(sum(parameter.numel() for parameter in wrapper.parameters()))
    if parameter_count != int(args.expected_parameters):
        raise RuntimeError(
            "parameter gate failed: {} != {}".format(
                parameter_count, int(args.expected_parameters)
            )
        )
    if getattr(base_network, "pure2d_cnn_context", None) is not None:
        raise RuntimeError("mainline unexpectedly constructed a CNN context module")
    checkpoint, loaded_tensor_count = strict_load(wrapper, checkpoint_path)
    checkpoint_step = int(checkpoint.get("step", -1))
    wrapper = wrapper.to(device).eval()
    memory_parameter_count = int(sum(
        parameter.numel() for name, parameter in wrapper.named_parameters()
        if "memory" in name.lower()
    ))
    if memory_parameter_count != 0:
        raise RuntimeError("Memory parameters unexpectedly exist")

    evaluator = make_evaluator(Evaluator, cfg, dataset, result_dir / "slice_metrics", _path_key)
    metric = load_module(metric_path, "pure2d_direct_verse2021")
    geometries = metric.prepare_dev8_geometry(slice_manifest, case_metadata, png_reader=None)
    collector = metric.PredictionVolumeCollector(geometries)
    teams_collector = None
    if teams_metric_path is not None:
        teams_metric = load_module(teams_metric_path, "pure2d_teams_style_2d")
        teams_collector = teams_metric.TEAMSStyleCollector(
            result_dir / "teams_style_2d",
            upstream_root=teams_upstream_root,
        )
    amp_context = (
        lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        if bool(cfg.use_amp) else nullcontext()
    )

    start = time.perf_counter()
    processed = 0
    seed_all(args.seed)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with amp_context():
                output = wrapper.net(batch["inp"], batch)
            masks = prediction_masks(
                output, batch, evaluator, inverse_affine_points, snake_config
            )
            paths = batch["img_path"]
            if isinstance(paths, str):
                paths = [paths]
            for image_path, mask in zip(paths, masks):
                record = evaluator._record_for_path(image_path)
                collector.add(record["case_id"], record["slice_idx"], mask)
                if teams_collector is not None:
                    teams_collector.add(
                        case_id=record["case_id"],
                        slice_idx=record["slice_idx"],
                        image_path=record["image_path"],
                        mask_path=record["mask_path"],
                        pred=mask,
                        gt=evaluator._read_mask(record["mask_path"]),
                    )
            evaluator.evaluate(output, batch)
            processed += len(masks)
            if processed % 160 < len(masks):
                print(
                    "[direct-eval] {}/{} slices".format(
                        processed, expected_processed
                    ),
                    flush=True,
                )
            del output, batch, masks
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if processed != expected_processed:
        raise RuntimeError("processed slice count drift")
    slice_summary = evaluator.summarize()
    teams_summary = teams_collector.finalize() if teams_collector is not None else None

    scan_results = []
    all_per_label = []
    verse_summary = None
    if not smoke_mode:
        collector.assert_complete()
        for case_id in DEV8_CASES:
            info = geometries[case_id]
            prediction = collector.source_volume(case_id)
            gt_volume = metric.load_nifti(info["gt_path"], info["gt_header"])
            result = metric.evaluate_scan(
                case_id,
                gt_volume,
                prediction,
                info["gt_header"].affine,
                info["centroids"],
            )
            scan_results.append(result)
            all_per_label.extend(result["per_label"])
        verse_summary = metric.summarize_cohort(scan_results)
        dump_json(result_dir / "verse2021_cohort.json", verse_summary)
        dump_json(
            result_dir / "verse2021_per_scan.json",
            [item["scan"] for item in scan_results],
        )
        dump_json(
            result_dir / "verse2021_per_vertebra.json", all_per_label
        )

    checkpoint_tag = (
        "step{}".format(checkpoint_step)
        if checkpoint_step >= 0
        else checkpoint_path.stem
    )
    visualization_path = result_dir / "routeb_{}_visualization.png".format(
        checkpoint_tag
    )
    visualization_rows = []
    if any(int(row["gt_foreground_pixels"]) > 0 for row in evaluator.results):
        visualization_rows = make_visualization(
            visualization_path,
            evaluator,
            collector,
            get_quadrangle,
            get_octagon,
        )
    else:
        visualization_path = None

    result = {
        "schema": "diffusionsnake.mainline.inference.v1",
        "status": "SMOKE_PASS" if smoke_mode else "PASS",
        "scope": "smoke" if smoke_mode else "complete_dev8",
        "reportable": not smoke_mode,
        "identity": {
            "project_root": str(project_root),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "runner_path": str(pathlib.Path(__file__).resolve()),
            "runner_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "metric_module_path": str(metric_path),
            "metric_module_sha256": sha256_file(metric_path),
            "teams_metric_module_path": (
                str(teams_metric_path) if teams_metric_path is not None else None
            ),
            "teams_metric_module_sha256": (
                sha256_file(teams_metric_path) if teams_metric_path is not None else None
            ),
            "teams_upstream_root": (
                str(teams_upstream_root) if teams_upstream_root is not None else None
            ),
            "seed": int(args.seed),
            "physical_gpu": int(args.gpu),
            "gpu_idle_max_memory_mib": int(args.max_idle_memory_mib),
            "gpu_idle_checks": gpu_checks,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model": {
            "architecture": "L6/F256",
            "parameter_count": parameter_count,
            "memory_parameter_count": memory_parameter_count,
            "internal_detector_parameter_count": 0,
            "detector_backend": str(cfg.detector_backend),
            "feature_backbone": "offline MoonViT layer_18 cache",
            "native_input_resolution": [512, 512],
            "feature_output_resolution": [128, 128],
            "loaded_tensor_count": loaded_tensor_count,
            "strict_checkpoint_load": True,
            "ha_smoe": {
                "routing_unit": "whole_contour",
                "routed_blocks": [2, 4, 6],
                "experts": 4,
                "top_k": 2,
                "shared_dense_ffn": True,
                "output_head": "dense",
            },
        },
        "inference": {
            "box_source": "GT axis-aligned rectangle",
            "initialization": "Route B rectangle -> 12-point octagon -> 128-point uniform contour",
            "class_source": "GT oracle",
            "solver": "AB2",
            "outer_stages": 2,
            "inner_steps_per_stage": 4,
            "nfe": 8,
            "model_forward_passes": 1,
            "memory_wrapper_constructed": False,
            "detector_constructed": False,
            "single_slice_only": True,
            "smoke_selection": (
                {
                    "mode": "first_foreground_rows_in_dev8_manifest_order",
                    "dataset_indices": smoke_indices,
                }
                if smoke_mode else None
            ),
            "batch_size": int(args.batch_size),
            "processed_slices": processed,
            "elapsed_seconds": elapsed,
            "slices_per_second": float(processed / elapsed),
        },
        "cohort": {
            "case_ids": sorted(
                {str(row["case_id"]) for row in evaluator.results}
            ),
            "case_count": len(
                {str(row["case_id"]) for row in evaluator.results}
            ),
            "slice_count": processed,
            "expected_complete_slice_count": EXPECTED_SLICES,
            "locked_case_opens": 0,
        },
        "slice_metrics": slice_summary,
        "verse2021": verse_summary,
        "teams_style_2d": teams_summary,
        "visualization": {
            "path": (
                None if visualization_path is None else str(visualization_path)
            ),
            "sha256": (
                None if visualization_path is None
                else sha256_file(visualization_path)
            ),
            "selected_slices": visualization_rows,
        },
    }
    result_path = result_dir / (
        "SMOKE_RESULTS.json"
        if smoke_mode
        else "PURE2D_DETECTOR_FREE_{}_RESULTS.json".format(
            checkpoint_tag.upper()
        )
    )
    dump_json(result_path, result)
    print(json.dumps({
        "status": "SMOKE_PASS" if smoke_mode else "PASS",
        "result_path": str(result_path),
        "visualization_path": (
            None if visualization_path is None else str(visualization_path)
        ),
        "checkpoint_step": checkpoint_step,
        "scan_equal_dice_mean": (
            None if verse_summary is None
            else verse_summary["main"]["scan_equal_dice_mean"]
        ),
        "identification_rate_mean": (
            None if verse_summary is None
            else verse_summary["main"]["scan_equal_identification_rate_mean"]
        ),
        "teams_style_2d_summary": (
            teams_summary["class_agnostic_all_images_upstream_compat"]
            if teams_summary is not None else None
        ),
    }, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
