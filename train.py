import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from src.utils.config_utils import read_config, dataset_from_config, use_tsp_objective
from src.model.model import ANYCSP
from src.model.loss import reinforce_loss
from src.csp.csp_data import CSP_Data

from argparse import ArgumentParser
from tqdm import tqdm
import json
import os


torch.multiprocessing.set_sharing_strategy('file_system')


def get_linear_scheduler():
    training_steps = config['epochs'] * len(train_loader)
    decay = config['lr_decay']
    lr_fn = lambda step: max(1.0 - ((1.0 - decay) * (step / training_steps)), decay)
    scheduler = LambdaLR(opt, lr_lambda=lr_fn)
    return scheduler


def save_opt_states(model_dir):
    torch.save(
        {
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        },
        os.path.join(model_dir, 'opt_state_dict.pt')
    )


def load_opt_states(model_dir):
    state_dicts = torch.load(os.path.join(model_dir, 'opt_state_dict.pt'))
    opt.load_state_dict(state_dicts['opt_state_dict'])
    scheduler.load_state_dict(state_dicts['sched_state_dict'])
    scaler.load_state_dict(state_dicts['scaler_state_dict'])
    return opt, scaler


def mean_or_none(total, count):
    if count == 0:
        return None
    return float(total / count)


def results_file_path(results_dir, model_dir):
    os.makedirs(results_dir, exist_ok=True)
    model_name = os.path.normpath(model_dir).replace(os.sep, '_')
    if os.altsep is not None:
        model_name = model_name.replace(os.altsep, '_')
    model_name = model_name.strip('._') or 'model'
    return os.path.join(results_dir, f'{model_name}_epochs.json')


def load_epoch_history(results_path):
    if not os.path.exists(results_path):
        return []
    with open(results_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('epochs'), list):
        return data['epochs']
    return []


def save_epoch_history(results_path, metadata, epoch_history):
    with open(results_path, 'w') as f:
        json.dump(
            {
                'metadata': metadata,
                'epochs': epoch_history,
            },
            f,
            indent=4
        )


def train_epoch():
    model.train()
    unsat_list = []
    unsat_ratio_list = []
    solved_list = []
    total_loss = 0.0
    total_unsat = 0.0
    total_unsat_ratio = 0.0
    total_solved = 0.0
    total_objective = 0.0
    total_tour_cost = 0.0
    total_duplicate = 0.0
    total_missing_edge = 0.0
    total_count = 0
    start_step = model.global_step

    for data in tqdm(train_loader, total=len(train_loader), disable=args.no_bar, desc=f'Training Epoch {epoch+1}'):
        opt.zero_grad()
        data.to(device)

        with torch.cuda.amp.autocast():
            data = model(
                data,
                config['T_train'],
                return_log_probs=True,
                return_all_unsat=True,
                return_all_assignments=True
            )

            loss = reinforce_loss(data, config)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        best_unsat = data.best_num_unsat.view(-1)
        unsat_ratio = best_unsat / data.batch_num_cst.view(-1)
        solved = best_unsat == 0
        batch_count = best_unsat.numel()

        unsat_list.append(best_unsat.cpu())
        unsat_ratio_list.append(unsat_ratio.cpu())
        solved_list.append(solved.cpu())
        total_loss += float(loss.detach().cpu().item()) * batch_count
        total_unsat += float(best_unsat.float().sum().detach().cpu().item())
        total_unsat_ratio += float(unsat_ratio.float().sum().detach().cpu().item())
        total_solved += float(solved.float().sum().detach().cpu().item())
        if use_tsp_objective(config):
            total_objective += float(data.best_objective.float().sum().detach().cpu().item())
            total_tour_cost += float(data.best_tour_cost.float().sum().detach().cpu().item())
            total_duplicate += float(data.best_duplicate_penalty.float().sum().detach().cpu().item())
            total_missing_edge += float(data.best_missing_edge_penalty.float().sum().detach().cpu().item())
        total_count += batch_count

        if (model.global_step + 1) % args.logging_steps == 0:
            unsat = torch.cat(unsat_list, dim=0)
            unsat_ratio = torch.cat(unsat_ratio_list, dim=0)
            solved = torch.cat(solved_list, dim=0)
            logger.add_scalar('Train/Loss', loss.mean(), model.global_step)
            logger.add_scalar('Train/Solved_Ratio', solved.float().mean(), model.global_step)
            logger.add_scalar('Train/Unsat_Count', unsat.float().mean(), model.global_step)
            logger.add_scalar('Train/Unsat_Ratio', unsat_ratio.float().mean(), model.global_step)
            unsat_list = []
            unsat_ratio_list = []
            solved_list = []

        if (model.global_step + 1) % args.checkpoint_steps == 0:
            model.save_model(name=f'checkpoint_{model.global_step}')

        model.global_step += 1

    metrics = {
        'loss': mean_or_none(total_loss, total_count),
        'solved_ratio': mean_or_none(total_solved, total_count),
        'unsat_count': mean_or_none(total_unsat, total_count),
        'unsat_ratio': mean_or_none(total_unsat_ratio, total_count),
        'samples': int(total_count),
        'global_step_start': int(start_step),
        'global_step_end': int(model.global_step),
        'steps': int(model.global_step - start_step),
    }
    if use_tsp_objective(config):
        metrics.update(
            {
                'tsp_objective': mean_or_none(total_objective, total_count),
                'tsp_tour_cost': mean_or_none(total_tour_cost, total_count),
                'tsp_duplicate_penalty': mean_or_none(total_duplicate, total_count),
                'tsp_missing_edge_penalty': mean_or_none(total_missing_edge, total_count),
            }
        )
    return metrics


def validate():
    model.eval()

    total_unsat = 0
    total_solved = 0
    total_objective = 0
    total_tour_cost = 0
    total_duplicate = 0
    total_missing_edge = 0
    total_count = 0

    for data in tqdm(val_loader, disable=args.no_bar, desc=f'Validating'):
        data.to(device)
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                data = model(
                    data,
                    config['T_val'],
                    return_log_probs=False,
                    return_all_unsat=True,
                    return_all_assignments=False
                )

        best_unsat = data.best_num_unsat.view(-1)
        total_unsat += best_unsat.float().sum().cpu().numpy()
        total_solved += (best_unsat == 0).float().sum().cpu().numpy()
        if use_tsp_objective(config):
            total_objective += data.best_objective.float().sum().cpu().numpy()
            total_tour_cost += data.best_tour_cost.float().sum().cpu().numpy()
            total_duplicate += data.best_duplicate_penalty.float().sum().cpu().numpy()
            total_missing_edge += data.best_missing_edge_penalty.float().sum().cpu().numpy()
        total_count += data.batch_size

    unsat = total_unsat / total_count
    solved = total_solved / total_count
    logger.add_scalar('Val/Solved_Ratio', solved, model.global_step)
    logger.add_scalar('Val/Unsat_Count', unsat, model.global_step)
    if use_tsp_objective(config):
        objective = total_objective / total_count
        tour_cost = total_tour_cost / total_count
        duplicate = total_duplicate / total_count
        missing_edge = total_missing_edge / total_count
        logger.add_scalar('Val/TSP_Objective', objective, model.global_step)
        logger.add_scalar('Val/TSP_Tour_Cost', tour_cost, model.global_step)
        logger.add_scalar('Val/TSP_Duplicate_Penalty', duplicate, model.global_step)
        logger.add_scalar('Val/TSP_Missing_Edge_Penalty', missing_edge, model.global_step)
        return unsat, solved, objective, tour_cost, duplicate, missing_edge
    return unsat, solved


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/comb/test', help="Model directory")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--logging_steps", type=int, default=10, help="Training steps between logging")
    parser.add_argument("--checkpoint_steps", type=int, default=5000, help="Training steps between saving checkpoints")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--no_bar", action='store_true', default=False, help="Turn of tqdm bar")
    parser.add_argument("--from_last", action='store_true', default=False, help="Continue from existing last checkpoint")
    parser.add_argument("--pretrained_dir", type=str, default=None, help="Pretrained Model directory")
    parser.add_argument("--config", type=str, help="path the config file")
    parser.add_argument("--results_dir", type=str, default='results', help="Directory for per-epoch training JSON logs")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.from_last:
        args.pretrained_dir = args.model_dir
        args.config = os.path.join(args.model_dir, 'config.json')

    config = read_config(args.config)

    if args.pretrained_dir is None:
        model = ANYCSP(args.model_dir, config)
    else:
        model = ANYCSP.load(args.pretrained_dir, f'last')
        model.model_dir = args.model_dir

    model.to(device)
    model.train()

    train_data = dataset_from_config(config['train_data'], config['epoch_steps'] * config['batch_size'])
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        collate_fn=CSP_Data.collate
    )

    if 'val_data' in config:
        val_data = dataset_from_config(config['val_data'])
        val_loader = DataLoader(
            val_data,
            batch_size=config['val_batch_size'],
            num_workers=args.num_workers,
            collate_fn=CSP_Data.collate
        )
    else:
        val_loader = None

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_linear_scheduler()
    scaler = torch.cuda.amp.GradScaler()

    if args.pretrained_dir is not None:
        opt, scaler = load_opt_states(args.pretrained_dir)

    logger = SummaryWriter(args.model_dir)
    results_path = results_file_path(args.results_dir, args.model_dir)
    results_metadata = {
        'model_dir': args.model_dir,
        'config': args.config,
        'seed': int(args.seed),
        'epochs': int(config['epochs']),
        'epoch_steps': int(config['epoch_steps']),
        'batch_size': int(config['batch_size']),
        'pretrained_dir': args.pretrained_dir,
    }
    epoch_history = load_epoch_history(results_path) if args.pretrained_dir is not None else []
    save_epoch_history(results_path, results_metadata, epoch_history)
    best_unsat = np.float32('inf')
    best_solved = 0.0
    start_step = 0
    for epoch in range(config['epochs']):
        train_metrics = train_epoch()
        epoch_record = {
            'epoch': int(len(epoch_history) + 1),
            'train': train_metrics,
        }
        epoch_history.append(epoch_record)
        save_epoch_history(results_path, results_metadata, epoch_history)

        if val_loader is not None:
            val_metrics = validate()
            unsat, solved = val_metrics[:2]
            epoch_record['validation'] = {
                'unsat_count': float(unsat),
                'solved_ratio': float(solved),
            }

            if use_tsp_objective(config):
                objective, tour_cost, duplicate, missing_edge = val_metrics[2:]
                epoch_record['validation'].update(
                    {
                        'tsp_objective': float(objective),
                        'tsp_tour_cost': float(tour_cost),
                        'tsp_duplicate_penalty': float(duplicate),
                        'tsp_missing_edge_penalty': float(missing_edge),
                    }
                )
                print(
                    f'Mean Unsat Count: {unsat:.2f}, Solved: {100 * solved:.2f}%, '
                    f'Mean TSP Objective: {objective:.2f}, Mean Tour Cost: {tour_cost:.2f}, '
                    f'Duplicates: {duplicate:.2f}, Missing Edges: {missing_edge:.2f}'
                )
                best_metric = objective
            else:
                print(f'Mean Unsat Count: {unsat:.2f}, Solved: {100 * solved:.2f}%')
                best_metric = unsat

            if best_metric < best_unsat:
                model.save_model(name='best')
                best_unsat = best_metric
                epoch_record['saved_best'] = True
            else:
                epoch_record['saved_best'] = False

        model.save_model(name='last')
        save_opt_states(model.model_dir)
        save_epoch_history(results_path, results_metadata, epoch_history)
