import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch

import evaluate_tsp as evaluation


ROOT = Path(__file__).resolve().parents[1]


def trajectory():
    # The objective prefers an invalid assignment; feasible selection must still
    # return the cheaper of two valid tours. The last column ties the initial one.
    return SimpleNamespace(
        all_assignments=torch.tensor([[0, 0, 0, 0], [0, 1, 2, 0], [2, 2, 1, 2]]),
        all_objective=torch.tensor([[4., 10., 8., 4.]]),
        all_tour_cost=torch.tensor([[2., 10., 8., 2.]]),
        all_duplicate_penalty=torch.tensor([[1., 0., 0., 1.]]),
        all_missing_edge_penalty=torch.tensor([[1., 0., 0., 1.]]),
        all_num_unsat=torch.tensor([[2., 0., 0., 2.]]),
        constraints={'tsp_edge': SimpleNamespace(
            src_var_idx=torch.tensor([0, 1, 2]),
            dst_var_idx=torch.tensor([1, 2, 0]),
        )},
    )


class SelectionTests(unittest.TestCase):
    def test_feasibility_and_objective_selection_are_distinct_with_earliest_ties(self):
        result = evaluation.summarize_trajectory(trajectory())
        self.assertEqual(result['best_objective']['step'], 0)
        self.assertEqual(result['best_objective']['assignment'], [0, 0, 2])
        self.assertEqual(result['best_objective']['self_loops'], 1)
        self.assertIsNone(result['best_objective']['valid_tour_cost'])
        self.assertEqual(result['best_feasible']['step'], 2)
        self.assertEqual(result['best_feasible']['assignment'], [0, 2, 1])
        self.assertEqual(result['best_feasible']['valid_tour_cost'], 8)
        self.assertEqual(result['first_feasible_step'], 1)
        self.assertEqual(result['feasible_steps'], 2)

    def test_no_valid_tours_have_null_costs_and_zero_success_rate(self):
        data = trajectory()
        data.all_assignments = data.all_assignments[:, [0, 3]]
        for attr in evaluation.METRICS.values():
            setattr(data, attr, getattr(data, attr)[:, [0, 3]])
        run = {'inference_seconds': 2., **evaluation.summarize_trajectory(data)}
        self.assertIsNone(run['best_feasible'])
        self.assertIsNone(run['first_feasible_step'])
        summary = evaluation.summarize_runs([run])
        self.assertEqual(summary['selected_feasible_ratio'], 0)
        self.assertEqual(summary['found_feasible_ratio'], 0)
        self.assertIsNone(summary['mean_valid_tour_cost'])
        self.assertIsNone(summary['best_feasible_tour_cost'])
        json.dumps(summary, allow_nan=False)

    def test_failed_atomic_write_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            evaluation.save_results(path, {'status': 'running', 'completed_runs': 1})
            previous = path.read_bytes()
            with patch.object(evaluation.json, 'dump', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    evaluation.save_results(path, {'completed_runs': 2})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])


class EvaluationIntegrationTests(unittest.TestCase):
    def test_real_n100_checkpoint_metrics_progress_and_no_overwrite(self):
        model_dir = ROOT / evaluation.DEFAULT_MODEL
        instance = sorted((ROOT / 'data/TravellingSalesman-m1-n100').glob('*.xml'))[0]
        checkpoint_hash = evaluation.sha256(model_dir / 'best.pkl')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'evaluation.json'
            args = [
                '--model_dir', str(model_dir), '--data_path', str(instance),
                '--network_steps', '2', '--seeds', '0', '1', '--device', 'cpu',
                '--output', str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(args)
            payload = json.loads(output.read_text())
            self.assertEqual(payload['status'], 'completed')
            self.assertEqual(payload['metadata']['checkpoint_sha256'], checkpoint_hash)
            self.assertEqual(payload['summary']['completed_runs'], 2)
            self.assertEqual(payload['summary']['completed_instances'], 1)
            self.assertEqual(payload['metadata']['instance_files'], [str(instance)])

            # Recompute against the XML rather than trusting the model's metrics.
            supports = ET.parse(instance).getroot().find('constraints/group/extension/supports').text
            edges = {(int(a), int(b)): int(cost) for a, b, cost in re.findall(
                r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', supports,
            )}
            for run in payload['instances'][0]['runs']:
                self.assertEqual(run['num_cities'], 100)
                self.assertEqual(run['network_steps'], 2)
                for key in ('best_objective', 'best_feasible'):
                    result = run[key]
                    if result is None:
                        continue
                    route = result['assignment']
                    self.assertEqual(len(route), 100)
                    self.assertTrue(all(0 <= city < 100 for city in route))
                    pairs = list(zip(route, route[1:] + route[:1]))
                    duplicate = 100 - len(set(route))
                    missing = sum(pair not in edges for pair in pairs)
                    cost = sum(edges.get(pair, 0) for pair in pairs)
                    self.assertEqual(result['tsp_duplicate_penalty'], duplicate)
                    self.assertEqual(result['tsp_missing_edge_penalty'], missing)
                    self.assertEqual(result['tsp_tour_cost'], cost)
                    self.assertEqual(result['self_loops'], sum(a == b for a, b in pairs))
                    self.assertEqual(result['tsp_objective'], (100 * max(edges.values()) + 1) * duplicate + cost)
                    self.assertEqual(result['valid_tour_cost'], cost if duplicate == missing == 0 else None)

            # Repeating a CPU seed must reproduce the result independently of
            # previously evaluated seeds; model weights must remain untouched.
            model = evaluation.load_checkpoint(model_dir, 'best', torch.device('cpu'))
            repeated = evaluation.evaluate_run(model, instance, 0, 2, torch.device('cpu'))
            original = payload['instances'][0]['runs'][0]
            self.assertEqual(repeated['best_objective'], original['best_objective'])
            self.assertEqual(evaluation.sha256(model_dir / 'best.pkl'), checkpoint_hash)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                evaluation.main(args)
            self.assertEqual(output.read_bytes(), before)

            # Preserve completed runs and mark partial results when inference fails.
            output.unlink()
            with patch.object(evaluation, 'evaluate_run', side_effect=[original, RuntimeError('inference failed')]):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'inference failed'):
                    evaluation.main(args)
            partial = json.loads(output.read_text())
            self.assertEqual(partial['status'], 'failed')
            self.assertEqual(partial['summary']['completed_runs'], 1)
            self.assertEqual(partial['summary']['completed_instances'], 0)
            self.assertEqual(len(partial['instances'][0]['runs']), 1)


if __name__ == '__main__':
    unittest.main()
