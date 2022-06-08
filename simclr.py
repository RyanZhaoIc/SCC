"""
References: https://github.com/wvangansbeke/Unsupervised-Classification.git
(Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/))
"""
import argparse
import os

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from scipy.spatial.distance import pdist, squareform
from termcolor import colored
from scan_utils.utils.common_config import get_criterion, get_model, get_train_dataset, \
    get_train_dataloader, \
    get_val_dataloader, get_train_transformations, \
    get_val_transformations, get_optimizer, \
    adjust_learning_rate
from scan_utils.utils.config import create_config
from scan_utils.utils.evaluate_utils import contrastive_evaluate
from scan_utils.utils.memory import MemoryBank
from scan_utils.utils.train_utils import simclr_train
from scan_utils.utils.utils import fill_memory_bank


def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    # state = torch.load(p['pretext_model'])
    # model.load_state_dict(state, strict=True)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Transform
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)

    # Split to facilitate the model selection of SCAN
    # All data will be used for training. Note that since the unlabelled validation set for scan model selection is
    # randomly divided from all training data, the experimental setup is not broken.
    # During the pretext training phase, all data from STL were used with reference to the SCAN setup.
    base_dataset = get_train_dataset(p, val_transforms, split='train+test')

    dataset_size = len(base_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_indices.sort()
    val_indices.sort()

    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True, split='train+test+unlabeled')
    base_scan_train_dataset = get_train_dataset(p, val_transforms, split='train+test',
                                                indices=train_indices)
    val_dataset = get_train_dataset(p, val_transforms, split='train+test', indices=val_indices)

    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # DataLoader
    print(colored('Build MemoryBank', 'blue'))
    base_dataloader = get_val_dataloader(p, base_dataset)
    train_dataloader = get_train_dataloader(p, train_dataset)
    base_scan_train_dataloader = get_val_dataloader(p, base_scan_train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)

    # Memory Bank
    memory_bank_base = MemoryBank(len(base_dataset),
                                  p['model_kwargs']['features_dim'],
                                  p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_scan_train = MemoryBank(len(base_scan_train_dataset),
                                        p['model_kwargs']['features_dim'],
                                        p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_scan_train.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                 p['model_kwargs']['features_dim'],
                                 p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' % (top1))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.module.state_dict(),
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

    # Save final model
    torch.save(model.module.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (SCC_Train)
    # These will be used for SCC.
    print(colored('Fill memory bank for mining the nearest neighbors (SCC_Train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = len(base_dataset) // p['num_classes']
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * acc))
    np.save(p['topk_neighbors_train_path'], indices)

    features = memory_bank_base.features.cpu().numpy()
    sq = squareform(pdist(features, 'euclidean'))
    np.save(p['distance_matrix_path'], sq)

    # Mine the topk nearest neighbors at the very end (Scan_Train)
    print(colored('Fill memory bank for mining the nearest neighbors (Scan_Train) ...', 'blue'))
    fill_memory_bank(base_scan_train_dataloader, model, memory_bank_scan_train)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices, acc = memory_bank_scan_train.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * acc))
    np.save(p['topk_neighbors_scan_train_path'], indices)

    # Mine the topk nearest neighbors at the very end (Scan_Val)
    print(colored('Fill memory bank for mining the nearest neighbors (Scan_Val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' % (topk, 100 * acc))
    np.save(p['topk_neighbors_scan_val_path'], indices)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='SimCLR')
    parser.add_argument('--config_env',
                        help='Config file for the environment', default='scan_utils/configs/env.yml')
    parser.add_argument('--config_exp',
                        help='Config file for the experiment', default='scan_utils/configs/pretext/simclr_cifar10.yml')
    parser.add_argument('--seed', default=5, type=int,
                        help='seed for initializing training. ')
    args = parser.parse_known_args()[0]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    main()
