import os
from timeit import default_timer as timer
import torch
from torch.nn import Module, GRUCell
from torch_scatter import scatter_min, scatter_sum

from src.model.layers import Val2Val_Layer, Cst2Val_Layer, Val2Cst_Layer, Policy
from src.utils.config_utils import get_tsp_objective_config, read_config, write_config


class ANYCSP(Module):

    def __init__(self, model_dir, config):
        super(ANYCSP, self).__init__()
        self.model_dir = model_dir
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.sampling = config['sampling']
        self.config.setdefault('fix_first_var', True)
        self.config.setdefault('weight_all_diff_reward', True)
        self.config.setdefault('initialize_all_diff', False)
        self.config['tsp_objective'] = get_tsp_objective_config(self.config)
        self.fix_first_var = self.config['fix_first_var']
        self.weight_all_diff_reward = self.config['weight_all_diff_reward']
        self.initialize_all_diff = self.config['initialize_all_diff']
        self.tsp_objective_config = self.config['tsp_objective']
        self.use_tsp_objective = self.tsp_objective_config['enabled']

        # GRU cell and its initial state
        self.h_val_init = torch.nn.Parameter(torch.normal(0.0, 1.0, (1, self.hidden_dim), dtype=torch.float32))
        self.val_cell = GRUCell(self.hidden_dim, self.hidden_dim)

        # module for msg. pass from values to constraints
        self.val2cst = Val2Cst_Layer(config)

        # module for msg. pass from constraints to values
        self.cst2val = Cst2Val_Layer(config)

        # module for msg. from values tor variables and back
        self.val2val = Val2Val_Layer(config)

        # Output mlp (O)
        self.policy = Policy(config)

        self.global_step = 0

    def save_model(self, model_dir=None, name='model'):
        if model_dir is None:
            model_dir = self.model_dir
        os.makedirs(model_dir, exist_ok=True)
        write_config(self.config, model_dir)
        state_dict = self.state_dict()
        state_dict['global_step'] = self.global_step
        torch.save(state_dict, os.path.join(model_dir, f'{name}.pkl'))

    @staticmethod
    def load_model(model_dir, name='model'):
        config = read_config(os.path.join(model_dir, 'config.json'))
        model = ANYCSP(model_dir, config)
        state_dict = torch.load(os.path.join(model_dir, f'{name}.pkl'))
        model.load_state_dict(state_dict, strict=False)
        model.global_step = state_dict['global_step']
        return model

    def init_fixed_first_var(self, data):
        if not self.fix_first_var:
            return None, None

        var_idx = torch.arange(data.num_var, device=data.device)
        fixed_var_idx = scatter_min(var_idx, data.batch, dim=0, dim_size=data.batch_size)[0]
        if self.initialize_all_diff:
            fixed_dom_idx = torch.zeros((data.batch_size,), dtype=torch.long, device=data.device)
        else:
            fixed_dom_idx = torch.floor(
                torch.rand((data.batch_size,), device=data.device) * data.domain_size[fixed_var_idx].float()
            ).long()
        fixed_val_idx = data.var_off[fixed_var_idx] + fixed_dom_idx

        data.fixed_var_idx = fixed_var_idx
        data.fixed_val_idx = fixed_val_idx
        data.fixed_dom_idx = fixed_dom_idx
        return fixed_var_idx, fixed_val_idx

    def init_all_diff_assignment(self, data):
        var_idx = torch.arange(data.num_var, device=data.device)
        first_var_idx = scatter_min(var_idx, data.batch, dim=0, dim_size=data.batch_size)[0]
        local_var_idx = var_idx - first_var_idx[data.batch]

        if torch.any(local_var_idx >= data.domain_size):
            raise ValueError('initialize_all_diff requires variable i to have value i in its domain.')

        val_idx = data.var_off + local_var_idx.long()
        assignment = torch.zeros((data.num_val, 1), dtype=torch.float32, device=data.device)
        assignment[val_idx] = 1.0
        return assignment

    def init_assignment(self, data, fixed_var_idx=None, fixed_val_idx=None):
        if self.initialize_all_diff:
            assignment = self.init_all_diff_assignment(data)
        else:
            logits = torch.ones((data.num_val,), device=data.device, dtype=torch.float32)
            assignment, _ = data.hard_assign_sample(logits, fixed_var_idx=fixed_var_idx, fixed_val_idx=fixed_val_idx)
        cst_sat = data.constraint_is_sat(assignment, update_LE=True)
        num_unsat = self.count_unsat(data, cst_sat)
        return assignment, num_unsat

    def update_assignment(self, data, logits, assignment, fixed_var_idx=None, fixed_val_idx=None):
        if self.sampling == 'local':
            assignment, log_prob = data.hard_assign_sample_local(logits, assignment, fixed_var_idx=fixed_var_idx)
        else:
            assignment, log_prob = data.hard_assign_sample(
                logits,
                fixed_var_idx=fixed_var_idx,
                fixed_val_idx=fixed_val_idx
            )
        cst_sat = data.constraint_is_sat(assignment, update_LE=True)
        num_unsat = self.count_unsat(data, cst_sat)
        return assignment, num_unsat, log_prob

    def count_unsat(self, data, cst_sat):
        unsat = 1.0 - cst_sat
        if data.cst_weight is not None:
            unsat = unsat * data.cst_weight.view(-1, 1)
        return scatter_sum(unsat, data.cst_batch, dim=0, dim_size=data.batch_size)

    @staticmethod
    def gather_metric(metric, idx):
        return metric.gather(1, idx.view(-1, 1)).view(-1)

    def init_tsp_tracking(self, data, assignment):
        metrics = data.tsp_objective(assignment, self.tsp_objective_config)
        objective, best_idx = metrics['objective'].min(dim=1)

        data.best_objective = objective
        data.best_tour_cost = self.gather_metric(metrics['tour_cost'], best_idx)
        data.best_duplicate_penalty = self.gather_metric(metrics['duplicate_penalty'], best_idx)
        data.best_missing_edge_penalty = self.gather_metric(metrics['missing_edge_penalty'], best_idx)
        data.tsp_big_m = metrics['big_m'].view(-1)

        data.all_objective = [objective.view(-1, 1)]
        data.all_tour_cost = [data.best_tour_cost.view(-1, 1)]
        data.all_duplicate_penalty = [data.best_duplicate_penalty.view(-1, 1)]
        data.all_missing_edge_penalty = [data.best_missing_edge_penalty.view(-1, 1)]
        return objective

    def update_tsp_tracking(self, data, assignment, num_unsat):
        metrics = data.tsp_objective(assignment, self.tsp_objective_config)
        objective, best_idx = metrics['objective'].min(dim=1)
        tour_cost = self.gather_metric(metrics['tour_cost'], best_idx)
        duplicate_penalty = self.gather_metric(metrics['duplicate_penalty'], best_idx)
        missing_edge_penalty = self.gather_metric(metrics['missing_edge_penalty'], best_idx)
        cur_num_unsat = self.gather_metric(num_unsat, best_idx)

        improved = objective < data.best_objective
        data.best_objective = torch.where(improved, objective, data.best_objective)
        data.best_tour_cost = torch.where(improved, tour_cost, data.best_tour_cost)
        data.best_duplicate_penalty = torch.where(improved, duplicate_penalty, data.best_duplicate_penalty)
        data.best_missing_edge_penalty = torch.where(improved, missing_edge_penalty, data.best_missing_edge_penalty)
        data.best_num_unsat = torch.where(improved, cur_num_unsat, data.best_num_unsat)

        data.all_objective.append(objective.view(-1, 1))
        data.all_tour_cost.append(tour_cost.view(-1, 1))
        data.all_duplicate_penalty.append(duplicate_penalty.view(-1, 1))
        data.all_missing_edge_penalty.append(missing_edge_penalty.view(-1, 1))
        return objective

    def forward(
            self,
            data,
            steps,
            stop_early=False,
            return_log_probs=False,
            return_all_assignments=False,
            return_all_unsat=False,
            verbose=False,
            keep_time=False,
            timeout=None
    ):
        data.init_adj()
        data.init_constraint_weights(weight_all_diff=self.weight_all_diff_reward)

        # initialize first assignment and states
        fixed_var_idx, fixed_val_idx = self.init_fixed_first_var(data)
        assignment, num_unsat = self.init_assignment(data, fixed_var_idx, fixed_val_idx)
        h_val = self.h_val_init.tile(data.num_val, 1)

        data.best_num_unsat = num_unsat.min(dim=1)[0]
        data.num_steps = 0
        if self.use_tsp_objective:
            self.init_tsp_tracking(data, assignment)

        value_assignment = data.domain[assignment.flatten().bool()]
        assignment_list = [value_assignment.view(-1, 1)]
        num_unsat_list = [data.best_num_unsat.view(-1, 1)]
        log_prob_list = []

        opt = data.best_objective.min() if self.use_tsp_objective else data.best_num_unsat.min()
        data.opt_step = 0
        if verbose:
            if self.use_tsp_objective:
                print(f'o {float(opt.detach().cpu()):.2f}')
            else:
                print(f'o {opt.int().cpu().numpy()}')

        keep_time |= timeout is not None
        if keep_time:
            start = timer()
            data.opt_time = 0.0

        for s in range(steps):
            # one round of mes passes
            r_cst, x_val = self.val2cst(data, h_val, assignment)
            y_val = self.cst2val(data, x_val, r_cst)
            z_val = self.val2val(data, y_val)

            # update states and predict logit scores
            h_val = self.val_cell(z_val, h_val)
            logits = self.policy(h_val)

            # sample next assignment
            assignment, num_unsat, log_prob = self.update_assignment(
                data,
                logits,
                assignment,
                fixed_var_idx,
                fixed_val_idx
            )
            data.num_steps = s + 1

            # update all kinds of metrics...
            if self.use_tsp_objective:
                objective = self.update_tsp_tracking(data, assignment, num_unsat)
                num_unsat = num_unsat.min(dim=1)[0]
                cur_opt = data.best_objective.min()
            else:
                num_unsat, best_assign = num_unsat.min(dim=1)
                data.best_num_unsat = torch.minimum(data.best_num_unsat, num_unsat)
                cur_opt = data.best_num_unsat.min()

            if return_log_probs:
                log_prob_list.append(log_prob.view(-1, 1))
            if return_all_unsat:
                num_unsat_list.append(num_unsat.view(-1, 1))
            if keep_time:
                cur = timer()
                time = float(cur - start)
            if cur_opt < opt:
                opt = cur_opt
                if verbose:
                    if self.use_tsp_objective:
                        print(f'o {float(opt.detach().cpu()):.2f}')
                    else:
                        print(f'o {opt.int().cpu().numpy()}')
                if keep_time:
                    data.opt_time = float(time)
                    data.opt_step = s + 1
            if stop_early and not self.use_tsp_objective and num_unsat.min() == 0.0:
                break
            if return_all_assignments:
                value_assignment = data.domain[assignment.flatten().bool()]
                assignment_list.append(value_assignment.view(-1, 1))
            if timeout is not None and timeout < time:
                break

        if return_log_probs:
            data.all_log_probs = torch.cat(log_prob_list, dim=1)
        if return_all_assignments:
            data.all_assignments = torch.cat(assignment_list, dim=1)
        if return_all_unsat:
            data.all_num_unsat = torch.cat(num_unsat_list, dim=1)
        if self.use_tsp_objective:
            data.all_objective = torch.cat(data.all_objective, dim=1)
            data.all_tour_cost = torch.cat(data.all_tour_cost, dim=1)
            data.all_duplicate_penalty = torch.cat(data.all_duplicate_penalty, dim=1)
            data.all_missing_edge_penalty = torch.cat(data.all_missing_edge_penalty, dim=1)
        if keep_time:
            data.total_time = time
        return data
