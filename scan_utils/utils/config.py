"""
References: https://github.com/wvangansbeke/Unsupervised-Classification.git
(Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/))
"""
import os
import yaml
from easydict import EasyDict
from ..utils.utils import mkdir_if_missing


def create_config(config_file_env, config_file_exp):
    # Config for environment path
    print(config_file_env)
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['pretext_acc'] = os.path.join(pretext_dir, 'acc.npy')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors_train+test.npy')
    cfg['topk_neighbors_scan_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors_train.npy')
    cfg['topk_neighbors_scan_val_path'] = os.path.join(pretext_dir, 'topk-train-neighbors_val.npy')
    cfg['distance_matrix_path'] = os.path.join(pretext_dir, 'distance_matrix_train+test.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel', 'reliability']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'scan')
        selflabel_dir = os.path.join(base_dir, 'selflabel')

        reliability_dir = os.path.join(base_dir, 'reliability')

        mkdir_if_missing(base_dir)
        mkdir_if_missing(scan_dir)
        mkdir_if_missing(selflabel_dir)

        mkdir_if_missing(reliability_dir)

        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
        cfg['scan_acc'] = os.path.join(scan_dir, 'acc.npy')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')
        cfg['selflabel_acc'] = os.path.join(selflabel_dir, 'acc.npy')
        cfg['reliability_dir'] = reliability_dir
        cfg['clean_ind_path'] = os.path.join(reliability_dir, 'clean_ind_v_ind.npy')
        cfg['pretrained_target_path'] = os.path.join(reliability_dir, 'pretrained_target.npy')

    return cfg


def pre_ssl_path(root_dir):
    base_dir = root_dir
    reliability_dir = os.path.join(base_dir, 'reliability')

    paths = EasyDict()
    paths['reliability_dir'] = reliability_dir
    paths['clean_ind_path'] = os.path.join(reliability_dir, 'clean_ind_v_ind.npy')
    paths['pretrained_target_path'] = os.path.join(reliability_dir, 'pretrained_target.npy')

    return paths

