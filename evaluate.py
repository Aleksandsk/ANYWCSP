import torch
import numpy as np
from argparse import ArgumentParser

from src.csp.csp_data import CSP_Data
from src.model.model import ANYCSP
from src.data.dataset import File_Dataset
from src.utils.config_utils import use_tsp_objective


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--checkpoint_name", type=str, default='best', help="Checkpoint to be used")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--network_steps", type=int, default=100000, help="Number of network steps during evaluation")
    parser.add_argument("--num_boost", type=int, default=1, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Output intermediate optima")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds")
    args = parser.parse_args()
    dict_args = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ANYCSP.load_model(args.model_dir, args.checkpoint_name)
    model.eval()
    model.to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = File_Dataset(args.data_path)

    num_solved = 0
    total_time = 0.0
    total_objective = 0.0
    total_tour_cost = 0.0
    num_total = len(dataset)

    for data in dataset:
        file = data.path

        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])

        data.to(device)

        if args.verbose:
            print(f'Solving {file}:')

        with torch.inference_mode():
            data = model(
                data,
                args.network_steps,
                return_all_assignments=False,
                return_log_probs=False,
                stop_early=True,
                return_all_unsat=False,
                verbose=args.verbose,
                keep_time=True,
                timeout=args.timeout
            )

        best_per_run = data.best_num_unsat.cpu().detach().numpy()
        if use_tsp_objective(model.config):
            objective_per_run = data.best_objective.cpu().detach().numpy()
            best_run = int(objective_per_run.argmin())
            best_objective = float(objective_per_run[best_run])
            best = float(best_per_run[best_run])
            best_tour_cost = float(data.best_tour_cost.cpu().detach().numpy()[best_run])
            best_duplicate = float(data.best_duplicate_penalty.cpu().detach().numpy()[best_run])
            best_missing_edge = float(data.best_missing_edge_penalty.cpu().detach().numpy()[best_run])
            total_objective += best_objective
            total_tour_cost += best_tour_cost
        else:
            best = best_per_run.min()
        solved = best == 0
        num_solved += int(solved)
        total_time += data.opt_time

        if use_tsp_objective(model.config):
            print(
                f'{file}: {"Solved" if solved else "Unsolved"}, '
                f'Num Unsat: {int(best)}, '
                f'TSP Objective: {best_objective:.2f}, '
                f'Tour Cost: {best_tour_cost:.2f}, '
                f'Duplicates: {best_duplicate:.2f}, '
                f'Missing Edges: {best_missing_edge:.2f}, '
                f'Steps: {data.num_steps}, '
                f'Opt Time: {data.opt_time:.2f}s, '
                f'Opt Step: {data.opt_step}'
            )
        else:
            print(
                f'{file}: {"Solved" if solved else "Unsolved"}, '
                f'Num Unsat: {int(best)}, '
                f'Steps: {data.num_steps}, '
                f'Opt Time: {data.opt_time:.2f}s, '
                f'Opt Step: {data.opt_step}'
            )

    if use_tsp_objective(model.config):
        print(
            f'Solved {100 * num_solved / num_total:.2f}%, '
            f'Average Objective: {total_objective / num_total:.2f}, '
            f'Average Tour Cost: {total_tour_cost / num_total:.2f}, '
            f'Average Time: {total_time / num_total:.2f}s'
        )
    else:
        print(f'Solved {100 * num_solved / num_total:.2f}%, Average Time: {total_time / num_total:.2f}s')
