"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from termcolor import colored
from scan_utils.utils.common_config import get_train_dataset, get_train_transformations, \
    get_val_transformations, \
    get_train_dataloader, get_val_dataloader, \
    get_optimizer, get_model, adjust_learning_rate, \
    get_criterion
from scan_utils.utils.config import create_config
from scan_utils.utils.ema import EMA
from scan_utils.utils.evaluate_utils import get_predictions, hungarian_evaluate
from scan_utils.utils.train_utils import selflabel_train


def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p, p['scan_model'])
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))

    # Transforms
    strong_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)

    # Split
    base_dataset = get_train_dataset(p, val_transforms, split='train+test')
    dataset_size = len(base_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_indices.sort()
    val_indices.sort()

    # Dataset
    train_dataset = get_train_dataset(p, {'standard': val_transforms, 'augment': strong_transforms},
                                      split='train+test', to_augmented_dataset=True, indices=train_indices)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataset = get_train_dataset(p, val_transforms, split='train+test', indices=val_indices)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(colored('Train samples %d - Val samples %d' % (len(train_dataset), len(val_dataset)), 'yellow'))

    # Checkpoint
    if os.path.exists(p['selflabel_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['selflabel_checkpoint']), 'blue'))
        checkpoint = torch.load(p['selflabel_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['selflabel_checkpoint']), 'blue'))
        start_epoch = 0

    # EMA
    if p['use_ema']:
        ema = EMA(model, alpha=p['ema_alpha'])
    else:
        ema = None

    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
        print(colored('-' * 10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Perform self-labeling
        print('Train ...')
        selflabel_train(train_dataloader, model, criterion, optimizer, epoch, ema=ema)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        predictions = get_predictions(p, val_dataloader, model)
        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
        print(clustering_stats)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['selflabel_checkpoint'])
        torch.save(model.module.state_dict(), p['selflabel_model'])

    # Evaluate and save the final model
    print(colored('Evaluate model at the end', 'blue'))
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(0, predictions,
                                          class_names=val_dataset.classes,
                                          compute_confusion_matrix=True,
                                          confusion_matrix_file=os.path.join(p['selflabel_dir'],
                                                                             'confusion_matrix.png'))
    print(clustering_stats)
    torch.save(model.module.state_dict(), p['selflabel_model'])


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Parser
    parser = argparse.ArgumentParser(description='SCAN Loss')
    parser.add_argument('--config_env',
                        help='Config file for the environment', default='scan_utils/configs/env.yml')
    parser.add_argument('--config_exp',
                        help='Config file for the experiment',
                        default='scan_utils/configs/selflabel/selflabel_cifar10.yml')
    parser.add_argument('--seed', default=5, type=int,
                        help='seed for initializing training. ')
    args = parser.parse_known_args()[0]

    main()
