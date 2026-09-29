#!/usr/bin/env python3
"""Fail-closed launcher for five-action Fourier GRPO and 2x4 deployment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.runtime import require_idle_gpu

PYTHON = Path(sys.executable).resolve()
TRAINER = REPO / 'tools/train_rl.py'
CONFIG = REPO / 'configs/stage2_rl.yaml'
SUPERVISED_CONFIG = REPO / 'configs/stage1.yaml'
REWARD_MODULE = REPO / 'lib/train/rewards/region_reward.py'
SOURCE = REPO / 'artifacts/checkpoints/ha_smoe_stage1_best.pt'
TUNE_MANIFEST = REPO / 'configs/manifests/volmem_fourier_validation37.csv'
PREFLIGHT = REPO / 'data/outputs/stage2_rl_preflight'
FORMAL = REPO / 'data/outputs/stage2_rl'
EXPECTED_SOURCE_SHA256 = ''
EXPECTED_TUNE_MANIFEST_SHA256 = (
    '24a4f19651edb5d187029f0255e2b59f9dce40f320ee29c14b709e8e92e6e6ad'
)
EXPECTED_SOURCE_STEP = -1
EXPECTED_MODEL_PARAMETERS = 16_933_968
EXPECTED_FLOW_PARAMETERS = 13_687_632
EXPECTED_CONTEXT_PARAMETERS = 3_246_336
FLOW_PREFIXES = ('net.gcn.', 'gcn.')


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def verify_inputs() -> dict:
    expected = {SOURCE: EXPECTED_SOURCE_SHA256, TUNE_MANIFEST: EXPECTED_TUNE_MANIFEST_SHA256}
    observed = {}
    for path in (
        TRAINER,
        CONFIG,
        SUPERVISED_CONFIG,
        REWARD_MODULE,
        SOURCE,
        TUNE_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_digest = sha256(path)
        observed[str(path)] = observed_digest
        expected_digest = expected.get(path)
        if expected_digest is not None and observed_digest != expected_digest:
            raise RuntimeError(
                f'SHA256 drift for {path}: {observed_digest} != {expected_digest}'
            )
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    observed[str(PYTHON)] = sha256(PYTHON)
    return observed


def state_dict(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ('state_dict', 'model', 'net', 'network'):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def audit_preflight(output: Path, launch_log: Path) -> dict:
    import torch

    latest = output / 'checkpoints/latest.pt'
    log_path = output / 'rl/logs.jsonl'
    hparams_path = output / 'rl/hparams.json'
    baseline_path = (
        output / 'rl/eval_baseline_step0.json'
    )
    required = (latest, log_path, hparams_path, baseline_path, launch_log)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f'preflight artifacts incomplete: {missing}')

    console = launch_log.read_text(encoding='utf-8', errors='replace')
    if (
        'five-stage AB2 training rollout alignment PASS max_abs=' not in console
        or 'production 2x4 AB2 deployment alignment PASS max_abs=' not in console
        or 'stage_s=[0.0, 0.6667]' not in console
    ):
        raise RuntimeError(
            'five-stage training and production 2x4 alignment evidence is missing'
        )

    rows = [
        json.loads(line) for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    if [int(row.get('step', -1)) for row in rows] != [1, 2]:
        raise RuntimeError('preflight must contain exactly effective steps 1 and 2')
    finite_keys = (
        'reward_mean', 'reward_std_mean', 'policy_loss', 'grad_norm',
        'ratio_mean', 'approx_kl', 'baseline_nsd_mean', 'final_nsd_mean',
        'step_quality_std_mean', 'step_burr_penalty_mean',
    )
    for row in rows:
        for key in finite_keys:
            if not math.isfinite(float(row[key])):
                raise RuntimeError(
                    f'non-finite metric step={row["step"]} {key}={row[key]}'
                )
        if row.get('action_policy') != 'geom':
            raise RuntimeError('wrong action policy')
        if row.get('reward_mode') != 'delta_nsd@2px':
            raise RuntimeError('wrong reward mode')
        if int(row.get('geom_lowfreq_modes', -1)) != 8:
            raise RuntimeError('wrong Fourier mode count')
        if [round(float(x), 4) for x in row['geom_sigma_px']] != [
            0.8, 0.7, 0.6, 0.5, 0.4
        ]:
            raise RuntimeError('wrong Fourier sigma schedule')
        if abs(float(row.get('outer_log_count_mean', -1.0)) - 5.0) > 1e-6:
            raise RuntimeError('each rollout must log exactly five outer actions')

    hparams = json.loads(hparams_path.read_text())
    fourier = hparams.get('fourier_configuration', {})
    reward_mode = hparams.get('reward_mode', {})
    credit = hparams.get('credit_assignment', {})
    deployment = hparams.get('deployment_schedule', {})
    if (
        hparams.get('schema') != 'diffusionsnake.mainline.stage2_rl.v3'
        or fourier.get('profile')
        != 'five_stage_m8_sigma_080_070_060_050_040'
        or fourier.get('eval_dataset') != 'VolMemFourierVal37'
        or int(fourier.get('eval_panel_size', -1)) != 80
        or fourier.get('dev8_used_for_selection') is not False
        or fourier.get('eval_latent_policy')
        != 'fixed_per_panel_row_across_all_steps'
        or reward_mode.get('name') != 'delta_nsd'
        or reward_mode.get('delta_nsd') is not True
        or round(float(reward_mode.get('nsd_delta_px', -1.0)), 4) != 2.0
        or reward_mode.get('coordinate_space') != '2d_image_pixels'
        or reward_mode.get('deterministic_reference')
        != 'same_stage_same_latent_flow'
        or reward_mode.get('composite_region_score_used_for_policy') is not False
        or credit.get('stage_reward')
        != 'sampled_delta_nsd_minus_deterministic_delta_nsd_minus_sampled_burr'
        or [round(float(x), 4) for x in hparams.get('stage_s_values', [])]
        != [0.0, 0.2, 0.4, 0.6, 0.8]
        or int(deployment.get('outer_steps', -1)) != 2
        or [round(float(x), 4) for x in deployment.get('fractions', [])]
        != [0.6667, 1.0]
        or [round(float(x), 4) for x in deployment.get('stage_s_values', [])]
        != [0.0, 0.6667]
        or int(deployment.get('ode_steps_per_outer_stage', -1)) != 4
        or int(deployment.get('total_nfe', -1)) != 8
        or deployment.get('used_for_all_reported_evaluation') is not True
    ):
        raise RuntimeError('five-stage train / 2x4 deploy metadata drift')

    baseline = json.loads(baseline_path.read_text())
    if (
        int(baseline.get('eval_n', -1)) <= 0
        or float(baseline.get('eval_iou', -1.0)) < 0.45
        or float(baseline.get('eval_dice', -1.0)) < 0.60
        or not 0.0 <= float(baseline.get('eval_nsd_2px', -1.0)) <= 1.0
    ):
        raise RuntimeError(f'supervised baseline sanity gate failed: {baseline}')

    source_payload = torch.load(str(SOURCE), map_location='cpu', weights_only=False)
    output_payload = torch.load(str(latest), map_location='cpu', weights_only=False)
    if int(source_payload.get('step', -1)) != EXPECTED_SOURCE_STEP:
        raise RuntimeError('source step drift')
    if int(output_payload.get('step', -1)) != 2:
        raise RuntimeError('preflight checkpoint step is not 2')
    source_state = state_dict(source_payload)
    output_state = state_dict(output_payload)
    if set(source_state) != set(output_state):
        raise RuntimeError('checkpoint key set differs from source')

    changed_flow = []
    changed_frozen = []
    nonfinite = []
    for key, source_tensor in source_state.items():
        output_tensor = output_state[key]
        if not torch.is_tensor(source_tensor) or not torch.is_tensor(output_tensor):
            continue
        if tuple(source_tensor.shape) != tuple(output_tensor.shape):
            raise RuntimeError(f'tensor shape drift: {key}')
        if not torch.isfinite(output_tensor).all().item():
            nonfinite.append(key)
        if key.startswith(FLOW_PREFIXES):
            if not torch.equal(source_tensor, output_tensor):
                changed_flow.append(key)
        elif not torch.equal(source_tensor, output_tensor):
            changed_frozen.append(key)
    if nonfinite or changed_frozen or not changed_flow:
        raise RuntimeError(
            'state audit failed: nonfinite={} changed_frozen={} changed_flow={}'.format(
                nonfinite[:8], changed_frozen[:8], len(changed_flow)
            )
        )
    return {
        'latest_checkpoint': str(latest),
        'latest_checkpoint_sha256': sha256(latest),
        'effective_steps': [1, 2],
        'changed_flow_tensors': len(changed_flow),
        'changed_frozen_tensors': len(changed_frozen),
        'nonfinite_tensors': len(nonfinite),
        'five_stage_training_alignment_pass': True,
        'deployment_2x4_alignment_pass': True,
    }


def main() -> int:
    global PYTHON, CONFIG, SOURCE, PREFLIGHT, FORMAL
    global EXPECTED_SOURCE_SHA256, EXPECTED_SOURCE_STEP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--python', type=Path, default=PYTHON)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument(
        '--source-sha256',
        required=True,
        help='full SHA256 of the selected supervised HA-SMoE checkpoint',
    )
    parser.add_argument(
        '--source-step',
        type=int,
        required=True,
        help='effective supervised update stored in the source checkpoint',
    )
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--slice-manifest', type=Path, required=True)
    parser.add_argument('--moonvit-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight-output', type=Path, default=PREFLIGHT)
    parser.add_argument(
        '--resume-checkpoint',
        type=Path,
        help='optional complete RL checkpoint; resume always writes to a fresh output',
    )
    args = parser.parse_args()

    PYTHON = args.python.expanduser().resolve()
    CONFIG = args.config.expanduser().resolve()
    SOURCE = args.source_checkpoint.expanduser().resolve()
    EXPECTED_SOURCE_SHA256 = str(args.source_sha256).strip().lower()
    EXPECTED_SOURCE_STEP = int(args.source_step)
    if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_SOURCE_SHA256):
        raise ValueError('--source-sha256 must contain exactly 64 hexadecimal characters')
    if EXPECTED_SOURCE_STEP <= 0:
        raise ValueError('--source-step must be positive')
    data_root = args.data_root.expanduser().resolve()
    slice_manifest = args.slice_manifest.expanduser().resolve()
    moonvit_cache = args.moonvit_cache.expanduser().resolve()
    PREFLIGHT = args.preflight_output.expanduser().resolve()
    resume_checkpoint = (
        args.resume_checkpoint.expanduser().resolve()
        if args.resume_checkpoint is not None else None
    )
    default_output = PREFLIGHT if args.mode == 'preflight' else FORMAL
    output = (args.output or default_output).expanduser().resolve()
    if args.mode == 'preflight':
        PREFLIGHT = output
    else:
        FORMAL = output
    for path, label, directory in (
        (PYTHON, 'Python interpreter', False),
        (CONFIG, 'RL config', False),
        (SOURCE, 'source checkpoint', False),
        (data_root, 'dataset root', True),
        (slice_manifest, 'slice manifest', False),
        (moonvit_cache, 'MoonViT cache', True),
    ):
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            raise FileNotFoundError(f'{label} missing: {path}')
    if resume_checkpoint is not None:
        if args.mode != 'train':
            raise ValueError('--resume-checkpoint is valid only with --mode train')
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f'RL resume checkpoint missing: {resume_checkpoint}'
            )
    hashes = verify_inputs()
    if output.exists():
        raise FileExistsError(f'refusing to reuse output: {output}')
    if args.mode == 'train':
        preflight_manifest_path = PREFLIGHT / 'manifest.json'
        if not preflight_manifest_path.is_file():
            raise RuntimeError('formal training requires the completed preflight')
        preflight_manifest = json.loads(preflight_manifest_path.read_text())
        if (
            preflight_manifest.get('status') != 'COMPLETED'
            or not preflight_manifest.get('preflight_audit')
            or preflight_manifest.get('hashes') != hashes
        ):
            raise RuntimeError('preflight is missing, failed, or input hashes changed')

    gpu_checks = require_idle_gpu(args.gpu)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / 'manifest.json'
    launch_log = output.parent / f'{output.name}.launch.log'
    manifest = {
        'status': 'STARTED',
        'mode': args.mode,
        'pipeline': 'moonvit_ha_smoe_five_action_fourier_delta_nsd_grpo_2x4_deploy',
        'output': str(output),
        'gpu': args.gpu,
        'launcher_pid': os.getpid(),
        'started_at': now(),
        'hashes': hashes,
        'gpu_idle_checks': gpu_checks,
        'source_checkpoint': str(SOURCE),
        'source_step': EXPECTED_SOURCE_STEP,
        'source_sha256': EXPECTED_SOURCE_SHA256,
        'resume_checkpoint': (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        'resume_checkpoint_sha256': (
            None if resume_checkpoint is None else sha256(resume_checkpoint)
        ),
        'model_parameters': EXPECTED_MODEL_PARAMETERS,
        'flow_trainable_parameters': EXPECTED_FLOW_PARAMETERS,
        'frozen_context_parameters': EXPECTED_CONTEXT_PARAMETERS,
        'ha_smoe': {
            'routing_unit': 'whole_contour',
            'routed_blocks': [2, 4, 6],
            'num_experts': 4,
            'top_k': 2,
            'shared_dense_ffn': True,
            'output_head': 'dense',
        },
        'training_contract': {
            'outer_fractions': [0.2, 0.25, 0.3333, 0.5, 1.0],
            'inner_steps_per_outer_stage': 4,
            'solver': 'ab2',
            'total_training_nfe': 20,
            'rl_action_count': 5,
            'fourier_modes': 8,
            'fourier_sigma_px': [0.8, 0.7, 0.6, 0.5, 0.4],
            'full_extrap_credit_map': [0, 1, 2, 3, 4],
            'stage_progress_s': [0.0, 0.2, 0.4, 0.59998, 0.79999],
            'inner_ab2_evaluations_are_rl_actions': False,
        },
        'deployment_contract': {
            'outer_fractions': [0.6667, 1.0],
            'inner_steps_per_outer_stage': 4,
            'solver': 'ab2',
            'total_nfe': 8,
            'stage_progress_s': [0.0, 0.6667],
            'used_for_all_reported_evaluation': True,
            'five_stage_training_rollout_is_not_reported_as_deployment': True,
        },
        'reward_contract': {
            'mode': 'delta_nsd',
            'nsd_delta_px': 2.0,
            'coordinate_space': '2d_image_pixels',
            'deterministic_reference': 'same_stage_same_latent_flow',
            'terminal_and_full_extrap_use_same_metric': True,
            'composite_region_score_used_for_policy': False,
            'sampled_burr_penalty_weight': 0.06,
        },
    }
    atomic_json(manifest_path, manifest)

    environment = os.environ.copy()
    existing_pythonpath = environment.get('PYTHONPATH', '').strip()
    environment.update({
        'CUDA_VISIBLE_DEVICES': str(args.gpu),
        'PYTHONUNBUFFERED': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'RL_OUTPUT_DIR': str(output),
        'CKPT_PATH': str(SOURCE),
        'DIFFUSIONSNAKE_DATA_ROOT': str(data_root),
        'DIFFUSIONSNAKE_SLICE_MANIFEST': str(slice_manifest),
        'MOONVIT_CACHE_ROOT': str(moonvit_cache),
        'DIFFUSIONSNAKE_SOURCE_SHA256': EXPECTED_SOURCE_SHA256,
        'DIFFUSIONSNAKE_SOURCE_STEP': str(EXPECTED_SOURCE_STEP),
        'PYTHONPATH': (
            str(REPO) if not existing_pythonpath
            else f'{REPO}{os.pathsep}{existing_pythonpath}'
        ),
    })
    if args.mode == 'preflight':
        environment['RL_STEPS'] = '2'
    elif resume_checkpoint is not None:
        environment['RL_RESUME_CHECKPOINT'] = str(resume_checkpoint)
    command = [str(PYTHON), str(TRAINER), '--cfg_file', str(CONFIG)]
    try:
        with launch_log.open('x', encoding='utf-8') as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            manifest['trainer_pid'] = process.pid
            atomic_json(manifest_path, manifest)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f'trainer exited with code {return_code}')
        if args.mode == 'preflight':
            manifest['preflight_audit'] = audit_preflight(output, launch_log)
        else:
            latest = output / 'checkpoints/latest.pt'
            if not latest.is_file():
                raise RuntimeError('formal training ended without latest checkpoint')
            manifest['latest_checkpoint_sha256'] = sha256(latest)
        manifest['status'] = 'COMPLETED'
        manifest['completed_at'] = now()
        atomic_json(manifest_path, manifest)
        return 0
    except BaseException as error:
        manifest['status'] = 'FAILED'
        manifest['failed_at'] = now()
        manifest['error'] = f'{type(error).__name__}: {error}'
        atomic_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    sys.exit(main())
