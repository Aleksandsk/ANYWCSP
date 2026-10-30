import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.csp.csp_data import CSP_Data
from src.data.dataset import load_xcsp3
from src.utils.assignment_logging import (
    assignment_file_path,
    best_training_assignments,
    save_assignment_history,
)
from train import results_file_path


ROOT = Path(__file__).resolve().parents[1]


class AssignmentSelectionTests(unittest.TestCase):
    def make_batch(self):
        # Different instance sizes and best steps expose incorrect batch slicing.
        data = CSP_Data.collate([
            CSP_Data(2, 3, path=Path('data/first.xml')),
            CSP_Data(3, 3, path='data/second.xml'),
        ])
        data.all_assignments = torch.tensor([
            [0, 1, 2], [1, 2, 0],
            [2, 1, 0], [0, 1, 2], [1, 0, 2],
        ])
        data.all_objective = torch.tensor([[9., 4., 5.], [3., 3., 6.]])
        data.all_tour_cost = torch.tensor([[10., 20., 30.], [40., 50., 60.]])
        data.all_duplicate_penalty = torch.tensor([[2., 1., 0.], [0., 1., 2.]])
        data.all_missing_edge_penalty = torch.tensor([[0., 1., 2.], [2., 1., 0.]])
        data.all_num_unsat = torch.tensor([[3., 2., 1.], [1., 2., 3.]])
        return data

    def test_per_instance_objective_minimum_with_initialization_and_ties(self):
        data = self.make_batch()
        rng_before = torch.get_rng_state().clone()
        assignments_before = data.all_assignments.clone()
        records = best_training_assignments(data, instance_offset=4)

        self.assertEqual([r['instance'] for r in records], ['data/first.xml', 'data/second.xml'])
        self.assertEqual([r['instance_index'] for r in records], [4, 5])
        self.assertEqual([r['best_step'] for r in records], [1, 0])
        self.assertEqual([r['assignment'] for r in records], [[1, 2], [2, 0, 1]])
        self.assertEqual([r['tsp_objective'] for r in records], [4., 3.])
        self.assertEqual([r['tsp_tour_cost'] for r in records], [20., 40.])
        self.assertEqual([r['tsp_duplicate_penalty'] for r in records], [1., 0.])
        self.assertEqual([r['tsp_missing_edge_penalty'] for r in records], [1., 2.])
        self.assertEqual([r['unsat_count'] for r in records], [2., 1.])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(data.all_assignments, assignments_before))
        json.dumps(records)

    def test_instances_without_file_paths_keep_indices(self):
        data = self.make_batch()
        data.instance_paths = [None, None]
        records = best_training_assignments(data)
        self.assertEqual([r['instance'] for r in records], [None, None])
        self.assertEqual([r['instance_index'] for r in records], [0, 1])


class AssignmentHistoryTests(unittest.TestCase):
    def test_path_keeps_results_directory_and_run_identifier(self):
        self.assertEqual(
            assignment_file_path('results/models_TSP_duplicates_tour_19000_945443_epochs.json'),
            'results/assigment_models_TSP_duplicates_tour_19000_945443_epochs.json',
        )

    def test_history_preserved_and_failed_write_leaves_previous_json_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assigment_run.json'
            metadata = {'interval_epochs': 10}
            epochs = [{'epoch': 10, 'instances': [{'assignment': [0, 1, 2]}]}]
            save_assignment_history(path, metadata, epochs)
            epochs.append({'epoch': 20, 'instances': [{'assignment': [0, 2, 1]}]})
            save_assignment_history(path, metadata, epochs)
            self.assertEqual(json.loads(path.read_text()), {'metadata': metadata, 'epochs': epochs})
            previous = path.read_bytes()

            with patch('src.utils.assignment_logging.json.dump', side_effect=OSError('write failed')):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    save_assignment_history(path, metadata, [])
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])


class TrainingAssignmentIntegrationTests(unittest.TestCase):
    def test_training_records_every_instance_only_at_epochs_10_and_20(self):
        with tempfile.TemporaryDirectory(prefix='anywcsp-assignment-test-') as directory:
            scratch = Path(directory)
            data_dir = scratch / 'data'
            data_dir.mkdir()
            sources = sorted((ROOT / 'data/TravellingSalesman-m1-n15').glob('*.xml'))[:3]
            self.assertEqual(len(sources), 3)
            for source in sources:
                shutil.copyfile(source, data_dir / source.name)

            config = json.loads((ROOT / 'configs/TSP/TSP_duplicates_tour_19000.json').read_text())
            config.update(epochs=21, T_train=2, T_val=2, batch_size=2, val_batch_size=1)
            config['train_data'] = {'FILES': {'path': str(data_dir / '*.xml')}}
            config['val_data'] = {'FILES': {'path': str(data_dir / sources[0].name)}}
            config_path = scratch / 'config.json'
            config_path.write_text(json.dumps(config))
            model_dir = scratch / 'models/TSP_duplicates_tour_19000_test'
            results_dir = scratch / 'results'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            completed = subprocess.run(
                [sys.executable, '-u', str(ROOT / 'train.py'), '--config', str(config_path),
                 '--model_dir', str(model_dir), '--results_dir', str(results_dir),
                 '--num_workers', '0', '--no_bar'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            results_path = results_file_path(str(results_dir), str(model_dir))
            results = json.loads(Path(results_path).read_text())
            snapshots = json.loads(Path(assignment_file_path(results_path)).read_text())
            self.assertEqual(len(results['epochs']), 21)
            self.assertEqual([record['epoch'] for record in snapshots['epochs']], [10, 20])
            self.assertEqual(snapshots['metadata']['tsp_objective'], config['tsp_objective'])
            self.assertEqual(snapshots['metadata']['interval_epochs'], 10)
            expected_paths = {str(data_dir / source.name) for source in sources}
            for snapshot in snapshots['epochs']:
                records = snapshot['instances']
                self.assertEqual(len(records), 3)  # Includes the final, smaller batch.
                self.assertEqual({r['instance'] for r in records}, expected_paths)
                self.assertEqual([r['instance_index'] for r in records], [0, 1, 2])
                for record in records:
                    self.assertIn(record['best_step'], range(3))
                    sample = load_xcsp3(record['instance'])
                    values = torch.tensor(record['assignment'])
                    self.assertEqual(values.numel(), sample.num_var)
                    assignment = (sample.domain == values[sample.var_idx]).float().view(-1, 1)
                    metrics = sample.tsp_objective(assignment, config['tsp_objective'])
                    for key in ('objective', 'tour_cost', 'duplicate_penalty', 'missing_edge_penalty'):
                        self.assertAlmostEqual(record[f'tsp_{key}'], metrics[key].item(), places=5)


if __name__ == '__main__':
    unittest.main()
