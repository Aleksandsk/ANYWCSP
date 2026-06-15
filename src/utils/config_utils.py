import os
import json
from src.data.dataset import File_Dataset, Generator_Dataset
from src.data.generators import generator_dict


def exists(path):
    path = os.path.join(path, 'config.json')
    return os.path.exists(path)


def write_config(config, path):
    path = os.path.join(path, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def read_config(path):
    with open(path, 'r') as f:
        conf_dict = json.load(f)
    return conf_dict


def get_tsp_objective_config(config):
    default = {
        'enabled': False,
        'use_duplicate_penalty': True,
        'use_missing_edge_penalty': True,
        'use_tour_cost': True,
    }

    objective_config = config.get('tsp_objective', None)
    if isinstance(objective_config, dict):
        default.update(objective_config)
    elif isinstance(objective_config, bool):
        default['enabled'] = objective_config
    else:
        default['enabled'] = bool(config.get('use_tsp_objective', False))
    return default


def use_tsp_objective(config):
    return get_tsp_objective_config(config)['enabled']


def dataset_from_config(data_config, num_samples=1000):
    if 'FILES' in data_config:
        dataset = File_Dataset(**data_config['FILES'])
    else:
        generators = [generator_dict[name](**kwargs) for name, kwargs in data_config.items()]
        dataset = Generator_Dataset(generators, num_samples)
    return dataset
