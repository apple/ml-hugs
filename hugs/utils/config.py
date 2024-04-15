#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import omegaconf
from itertools import product

from tqdm import tqdm


def flatten(dictionary, parent_key='', separator='/', dtype=dict):
    from collections.abc import MutableMapping
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dtype(items)


def unflatten(dictionary, separator='/', dtype=dict):
    resultDict = dtype()
    for key, value in dictionary.items():
        parts = key.split(separator)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dtype()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def get_cfg_items(cfg):
    cfg_file = flatten(cfg, separator='.')
    
    print(f'Experiment hyperparams:')
    
    hyperparam_search_keys = []
    for k in cfg_file.keys():
        if isinstance(cfg_file[k], omegaconf.listconfig.ListConfig):
            print(f'    {k}, vals={cfg_file[k]}')
            hyperparam_search_keys.append(k)
            
    cfg_search_results = product(*[cfg_file[k] for k in hyperparam_search_keys])
    list_of_cfgs = []

    for cfg_vals in tqdm(cfg_search_results):
        n_cfg_file = cfg_file.copy()
        for i, v in enumerate(cfg_vals):
            n_cfg_file[hyperparam_search_keys[i]] = v
            n_cfg_file['exp_name'] += f'-{hyperparam_search_keys[i]}={v}'
            
        list_of_cfgs.append(unflatten(n_cfg_file, separator='.'))
    
    list_of_cfgs = [omegaconf.OmegaConf.create(cfg) for cfg in list_of_cfgs]
    return list_of_cfgs, hyperparam_search_keys