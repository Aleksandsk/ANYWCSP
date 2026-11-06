"""Evaluate a saved TSP checkpoint and save assignments, metrics and progress."""

import argparse
from datetime import datetime, timezone
from glob import glob
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
from statistics import mean
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from src.data.dataset import load_xcsp3
from src.model.model import ANYCSP
from src.utils.config_utils import use_tsp_objective


DEFAULT_MODEL = 'models/TSP_duplicates_tour_19000_909674'
DEFAULT_DATA = 'data/TravellingSalesman-m1-n100/*.xml'
METRICS = {
    'tsp_objective': 'all_objective',
    'tsp_tour_cost': 'all_tour_cost',
    'tsp_duplicate_penalty': 'all_duplicate_penalty',
    'tsp_missing_edge_penalty': 'all_missing_edge_penalty',
    'unsat_count': 'all_num_unsat',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with open(path, 'rb') as source:
        return hashlib.sha256(source.read()).hexdigest()


def save_results(path, payload):
    """An interrupted write leaves the previously saved progress readable."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, indent=2, allow_nan=False)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(model_dir, checkpoint_name, device):
    config = json.loads((model_dir / 'config.json').read_text())
    if not use_tsp_objective(config):
        raise ValueError('This evaluator requires a checkpoint with the TSP objective enabled.')
    model = ANYCSP(str(model_dir), config)
    state = torch.load(
        model_dir / f'{checkpoint_name}.pkl', map_location='cpu', weights_only=True,
    )
    model.global_step = int(state.pop('global_step'))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def summarize_trajectory(data):
    """Select the earliest objective minimum and the cheapest valid tour separately."""
    histories = {
        name: getattr(data, attr).detach().cpu().view(-1)
        for name, attr in METRICS.items()
    }
    if not all(torch.isfinite(values).all() for values in histories.values()):
        raise ValueError('Non-finite evaluation metrics encountered.')
    assignments = data.all_assignments.detach().cpu()
    feasible = (
        (histories['tsp_duplicate_penalty'] == 0)
        & (histories['tsp_missing_edge_penalty'] == 0)
        & (histories['unsat_count'] == 0)
    )
    edges = data.constraints['tsp_edge']
    src, dst = edges.src_var_idx.cpu(), edges.dst_var_idx.cpu()

    def record(step):
        assignment = assignments[:, step]
        valid = bool(feasible[step])
        metrics = {name: float(values[step]) for name, values in histories.items()}
        return {
            'step': step,
            'assignment': assignment.tolist(),
            **metrics,
            'self_loops': int((assignment[src] == assignment[dst]).sum()),
            'feasible': valid,
            'valid_tour_cost': metrics['tsp_tour_cost'] if valid else None,
        }

    best_step = int(histories['tsp_objective'].argmin())
    feasible_steps = feasible.nonzero(as_tuple=True)[0]
    best_feasible_step = (
        int(feasible_steps[histories['tsp_tour_cost'][feasible_steps].argmin()])
        if len(feasible_steps) else None
    )
    return {
        'best_objective': record(best_step),
        'best_feasible': record(best_feasible_step) if best_feasible_step is not None else None,
        'first_feasible_step': int(feasible_steps[0]) if len(feasible_steps) else None,
        'feasible_steps': len(feasible_steps),
    }


def evaluate_run(model, instance, seed, network_steps, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # forward mutates the CSP; load a fresh instance for each independent seed.
    data = load_xcsp3(str(instance))
    if 'tsp_edge' not in data.constraints:
        raise ValueError(f'No TSP edge constraints found in {instance}')
    data.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = perf_counter()
    with torch.inference_mode():
        data = model(
            data, network_steps, stop_early=False,
            return_all_assignments=True, return_all_unsat=True,
            return_log_probs=False,
        )
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    return {
        'seed': seed,
        'num_cities': int(data.num_var),
        'network_steps': int(data.num_steps),
        'inference_seconds': elapsed,
        **summarize_trajectory(data),
    }


def summarize_runs(runs):
    selected = [r['best_objective'] for r in runs]
    valid = [r['valid_tour_cost'] for r in selected if r['feasible']]
    best_valid = [r['best_feasible']['valid_tour_cost'] for r in runs if r['best_feasible'] is not None]
    return {
        'completed_runs': len(runs),
        'selected_feasible_ratio': mean(r['feasible'] for r in selected) if runs else None,
        'found_feasible_ratio': len(best_valid) / len(runs) if runs else None,
        **{f'mean_{name}': mean(r[name] for r in selected) if runs else None for name in METRICS},
        'mean_self_loops': mean(r['self_loops'] for r in selected) if runs else None,
        'mean_valid_tour_cost': mean(valid) if valid else None,
        'mean_best_feasible_tour_cost': mean(best_valid) if best_valid else None,
        'best_feasible_tour_cost': min(best_valid) if best_valid else None,
        'mean_inference_seconds': mean(r['inference_seconds'] for r in runs) if runs else None,
        'total_inference_seconds': sum(r['inference_seconds'] for r in runs),
    }


def summarize_instances(instances, expected_runs_per_instance):
    runs = [r for instance in instances for r in instance['runs']]
    best_costs = [
        instance['summary']['best_feasible_tour_cost'] for instance in instances
        if instance['summary']['best_feasible_tour_cost'] is not None
    ]
    return {
        **summarize_runs(runs),
        'evaluated_instances': len(instances),
        'completed_instances': sum(len(i['runs']) == expected_runs_per_instance for i in instances),
        'instances_with_feasible_tour': len(best_costs),
        'instance_solved_ratio': len(best_costs) / len(instances) if instances else None,
        'mean_best_feasible_cost_per_instance': mean(best_costs) if best_costs else None,
    }


def environment_metadata(device):
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        'git_commit': commit,
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ('torch', 'numpy', 'torch-geometric', 'torch-scatter')},
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'precision': 'float32, autocast disabled',
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'hostname': platform.node(),
    }


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument('--data_path', default=DEFAULT_DATA, help='XML file or quoted glob pattern')
    parser.add_argument('--checkpoint_name', default='best', help='Checkpoint basename without .pkl')
    parser.add_argument('--network_steps', type=positive_int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path, help='New output JSON; existing files are never overwritten')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must be distinct integers in [0, 2**32).')
    files = sorted(Path(p) for p in glob(args.data_path))
    if not files or any(not p.is_file() or p.suffix != '.xml' for p in files):
        parser.error(f'Expected XML instances matching {args.data_path}')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    device = torch.device(device_name)
    checkpoint = args.model_dir / f'{args.checkpoint_name}.pkl'
    output = args.output or Path('results') / (
        f'evaluation_{args.model_dir.name}_{files[0].parent.name}_{args.checkpoint_name}_'
        f'{os.environ.get("SLURM_JOB_ID", "local")}_{uuid4().hex[:12]}.json'
    )
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    model = load_checkpoint(args.model_dir, args.checkpoint_name, device)
    payload = {
        'schema_version': 1,
        'status': 'running',
        'metadata': {
            'created_utc': utc_now(),
            'model_dir': str(args.model_dir),
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint),
            'checkpoint_global_step': model.global_step,
            'model_config': model.config,
            'data_path': args.data_path,
            'instance_files': [str(p) for p in files],
            'expected_instances': len(files),
            'expected_runs': len(files) * len(args.seeds),
            'network_steps': args.network_steps,
            'seeds': args.seeds,
            'selection': 'Minimum configured TSP objective, earliest step on ties; step 0 is initialization.',
            'assignment_order': 'CSP variable order, original domain values; the closing edge is included in costs.',
            'cost_semantics': 'tsp_tour_cost is raw cost, even for invalid assignments; valid_tour_cost is null when infeasible.',
            'timing': 'Synchronized forward inference only; excludes XML parsing, transfer and JSON writes.',
            'environment': environment_metadata(device),
        },
        'instances': [],
        'summary': summarize_instances([], len(args.seeds)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the name exclusively, including when another process races this one.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(f'Results: {output}', flush=True)
    print(f'{len(files)} instances x {len(args.seeds)} seeds x {args.network_steps} steps on {device}', flush=True)
    try:
        for path in files:
            instance = {'instance': str(path), 'sha256': sha256(path), 'runs': [], 'summary': summarize_runs([])}
            payload['instances'].append(instance)
            for seed in args.seeds:
                run = evaluate_run(model, path, seed, args.network_steps, device)
                instance['runs'].append(run)
                instance['summary'] = summarize_runs(instance['runs'])
                payload['summary'] = summarize_instances(payload['instances'], len(args.seeds))
                save_results(output, payload)
                best = run['best_objective']
                print(
                    f'{path.name} seed={seed}: objective={best["tsp_objective"]:.2f}, '
                    f'duplicates={best["tsp_duplicate_penalty"]:.0f}, '
                    f'missing={best["tsp_missing_edge_penalty"]:.0f}, '
                    f'valid_cost={best["valid_tour_cost"]}, {run["inference_seconds"]:.2f}s',
                    flush=True,
                )
        payload['status'] = 'completed'
        payload['metadata']['finished_utc'] = utc_now()
        save_results(output, payload)
    except Exception as exc:
        payload['status'] = 'failed'
        payload['error'] = f'{type(exc).__name__}: {exc}'
        save_results(output, payload)
        raise
    print(json.dumps(payload['summary'], indent=2), flush=True)
    return output


if __name__ == '__main__':
    main()
