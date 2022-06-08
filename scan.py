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
from scan_utils.utils.common_config import get_train_transformations, get_val_transformations, \
    get_train_dataset, get_train_dataloader, get_val_dataloader, get_optimizer, get_model, get_criterion, \
    adjust_learning_rate
from scan_utils.utils.config import create_config
from scan_utils.utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from scan_utils.utils.train_utils import scan_train


def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Transformations
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)

    # Split
    base_dataset = get_train_dataset(p, val_transformations, split='train+test')
    dataset_size = len(base_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_indices.sort()
    val_indices.sort()

    # Dataset
    print(colored('Get dataset and dataloaders', 'blue'))
    train_dataset = get_train_dataset(p, train_transformations, split='train+test',
                                      neighbor_path=p['topk_neighbors_scan_train_path'], indices=train_indices)
    val_dataset = get_train_dataset(p, val_transformations, split='train+test',
                                    neighbor_path=p['topk_neighbors_scan_val_path'], indices=val_indices)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' % (len(train_dataset), len(val_dataset)))

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # Checkpoint
    if os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']

    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None

    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        scan_train(train_dataloader, model, criterion, optimizer, epoch, p['update_cluster_head_only'])

        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)

        print('Evaluate based on SCAN loss ...')
        scan_stats = scan_evaluate(predictions)
        print(scan_stats)
        lowest_loss_head = scan_stats['lowest_loss_head']
        lowest_loss = scan_stats['lowest_loss']

        if lowest_loss < best_loss:
            print('New lowest loss on validation set: %.4f -> %.4f' % (best_loss, lowest_loss))
            print('Lowest loss head is %d' % (lowest_loss_head))
            best_loss = lowest_loss
            best_loss_head = lowest_loss_head
            torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model'])

        else:
            print('No new lowest loss on validation set: %.4f -> %.4f' % (best_loss, lowest_loss))
            print('Lowest loss head is %d' % (best_loss_head))

        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print(clustering_stats)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head},
                   p['scan_checkpoint'])

    # Evaluate and save the final model
    print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(model_checkpoint['head'], predictions,
                                          class_names=val_dataset.dataset.classes,
                                          compute_confusion_matrix=True,
                                          confusion_matrix_file=os.path.join(p['scan_dir'], 'confusion_matrix.png'))
    print(clustering_stats)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description='SCAN Loss')
    parser.add_argument('--config_env',
                        help='Config file for the environment', default='scan_utils/configs/env.yml')
    parser.add_argument('--config_exp',
                        help='Config file for the experiment', default='scan_utils/configs/scan/scan_cifar10.yml')
    parser.add_argument('--seed', default=5, type=int,
                        help='seed for initializing training. ')
    args = parser.parse_known_args()[0]

    main()
