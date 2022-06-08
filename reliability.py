# %%

import argparse
import copy
import os

import numpy as np
import torch.backends.cudnn
from termcolor import colored

from scan_utils.utils import utils
from scan_utils.utils.common_config import get_val_transformations, get_train_dataset, get_model, get_val_dataloader
from scan_utils.utils.config import create_config
from scan_utils.utils.evaluate_utils import get_predictions, hungarian_evaluate


def main():
    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    val_transformations = get_val_transformations(p)

    estimate_dataset = get_train_dataset(p, val_transformations, split='train+test')
    estimate_dataloader = get_val_dataloader(p, estimate_dataset)

    print('Validation transforms:', val_transformations)
    print('Data Number:', len(estimate_dataset))
    neighbors_simple = np.load(p['topk_neighbors_train_path'])

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['selflabel_model'])
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    predictions_es = get_predictions(p, estimate_dataloader, model)

    # Weight_Matrix
    distances = np.load(p['distance_matrix_path'])
    weight_matrix = np.empty_like(neighbors_simple[:, 1:]).astype(float)
    for i in range(len(neighbors_simple)):
        weight_matrix[i] = distances[i][neighbors_simple[i, 1:]]
    weight = ((weight_matrix.max() - weight_matrix) + 1e-8) / np.tile(
        (weight_matrix.max() - weight_matrix).sum(1) + 1e-8,
        (weight_matrix.shape[1], 1)).T

    # reliability
    neighbor_target_onehot = utils.one_hot(predictions_es[0]['predictions'])[neighbors_simple[:, 1:]]
    weight = weight.reshape([len(estimate_dataset), -1, 1])
    weight = weight.astype(np.float16)
    neighbor_target_onehot = neighbor_target_onehot.astype(np.int8)
    reliability = (weight * neighbor_target_onehot).sum(1) + predictions_es[0]['probabilities'].numpy().astype(np.float16)
    del neighbor_target_onehot

    h = args.h
    while h - 1 > 0:
        neighbor_reliability = reliability[neighbors_simple[:, 1:]]
        reliability = (weight * neighbor_reliability).sum(1) + predictions_es[0]['probabilities'].numpy().astype(
            np.float16)
        h -= 1

    # sample selection
    clean_ind_v_ind = []
    for i in range(p['num_classes']):
        clean_ind_v_ind.extend(
            torch.tensor(reliability[:, i], dtype=torch.float).argsort(descending=True)[
            :args.num_labels_per_class].tolist())

    match = hungarian_evaluate(0, predictions_es, compute_confusion_matrix=False)['hungarian_match']
    match = torch.tensor([i[1] for i in match])
    for i in range(p['num_classes']):
        predictions_es[0]['predictions'][predictions_es[0]['predictions'] == i] = match[i] + p['num_classes']
    predictions_es[0]['predictions'] -= p['num_classes']
    if p['train_db_name'] == 'stl-10':
        clean_ind_v_ind_es = copy.deepcopy(clean_ind_v_ind)
        clean_ind_v_ind_es = np.array(clean_ind_v_ind_es)
        clean_ind_v_ind_es = clean_ind_v_ind_es[clean_ind_v_ind_es < 5000]
        clean_ratio = (predictions_es[0]['predictions'][clean_ind_v_ind_es] == predictions_es[0]['targets'][
            clean_ind_v_ind_es]).sum().item() / len(clean_ind_v_ind_es)
    else:
        clean_ratio = (predictions_es[0]['predictions'][clean_ind_v_ind] == predictions_es[0]['targets'][
            clean_ind_v_ind]).sum().item() / len(clean_ind_v_ind)
    print(clean_ratio)
    print(clean_ind_v_ind)

    np.save(p['clean_ind_path'], clean_ind_v_ind)
    np.save(p['pretrained_target_path'], predictions_es[0]['predictions'].tolist())


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    FLAGS = argparse.ArgumentParser(description='SCAN Loss')
    FLAGS.add_argument('--config_env',
                       help='Config file for the environment', default='scan_utils/configs/env.yml')
    FLAGS.add_argument('--config_exp',
                       help='Config file for the experiment', default='scan_utils/configs/reliability/cifar10.yml')
    FLAGS.add_argument('--num_labels_per_class', type=int,
                       help='m', default=200)
    FLAGS.add_argument('--seed', default=5, type=int,
                       help='seed for initializing training. ')
    FLAGS.add_argument('--h', default=5, type=int, help='h. ')
    args = FLAGS.parse_known_args()[0]
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    main()
