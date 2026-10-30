"""Snapshots of the best TSP assignments already sampled during training."""

import json
import os
import tempfile


ASSIGNMENT_INTERVAL_EPOCHS = 10


def assignment_file_path(results_path):
    # Keep the requested "assigment_" spelling and the existing run identifier.
    return os.path.join(
        os.path.dirname(results_path), f'assigment_{os.path.basename(results_path)}'
    )


def best_training_assignments(data, instance_offset=0):
    """Select each instance's earliest minimum-objective rollout step, including 0.

    all_assignments contains domain values in CSP variable order. Instances in a
    batch can have different numbers of variables and different best steps.
    Reading these detached tensors does not sample again or change the loss.
    """
    best_steps = data.all_objective.detach().argmin(dim=1)
    assignments = data.all_assignments.detach().gather(
        1, best_steps[data.batch].view(-1, 1)
    ).view(-1).cpu()
    batch = data.batch.detach().cpu()
    metric_histories = {
        'tsp_objective': data.all_objective,
        'tsp_tour_cost': data.all_tour_cost,
        'tsp_duplicate_penalty': data.all_duplicate_penalty,
        'tsp_missing_edge_penalty': data.all_missing_edge_penalty,
        'unsat_count': data.all_num_unsat,
    }
    metrics = {
        name: values.detach().gather(1, best_steps.view(-1, 1)).view(-1).cpu().tolist()
        for name, values in metric_histories.items()
    }

    records = []
    for index, step in enumerate(best_steps.cpu().tolist()):
        path = data.instance_paths[index]
        records.append({
            'instance_index': instance_offset + index,
            'instance': os.fspath(path) if path is not None else None,
            'best_step': step,
            'assignment': assignments[batch == index].tolist(),
            **{name: values[index] for name, values in metrics.items()},
        })
    return records


def save_assignment_history(path, metadata, epochs):
    """Replace the JSON atomically so an interrupted write preserves old snapshots."""
    directory = os.path.dirname(path) or '.'
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=directory,
            prefix=f'.{os.path.basename(path)}.', suffix='.tmp', delete=False,
        ) as output:
            temporary_path = output.name
            json.dump({'metadata': metadata, 'epochs': epochs}, output, indent=2)
            output.write('\n')
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
