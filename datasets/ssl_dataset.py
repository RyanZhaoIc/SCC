import numpy as np
import torchvision
from torchvision import transforms

from .data_utils import split_ssl_data
from .dataset import BasicDataset
from scan_utils.utils.config import pre_ssl_path
from scan_utils.utils.mypath import MyPath

mean, std, crop_size, cut_size = {}, {}, {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['cifar20'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [0.447, 0.440, 0.406]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['cifar20'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['stl10'] = [0.260, 0.257, 0.270]

crop_size['cifar10'] = 32
crop_size['cifar100'] = 32
crop_size['cifar20'] = 32
crop_size['stl10'] = 96

cut_size['cifar10'] = 16
cut_size['cifar100'] = 16
cut_size['cifar20'] = 16
cut_size['stl10'] = 32


def get_transform(mean, std, train=True, crop_size=32):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(crop_size, padding=int(crop_size * 0.125),
                                                         padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir=MyPath.db_root_dir(),
                 save_dir='./results',
                 auxiliary=False):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """

        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform(mean[name], std[name], train, crop_size[name])
        self.paths = pre_ssl_path(save_dir)
        self.auxiliary = auxiliary

    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.name in ['cifar10', 'cifar100']:
            dset = getattr(torchvision.datasets, self.name.upper())
            dset_test = dset(self.data_dir, train=False, download=True)
            dset = dset(self.data_dir, train=True, download=True)
            data, targets = np.concatenate([dset.data, dset_test.data], axis=0), (dset.targets + dset_test.targets)
        elif self.name == 'cifar20':
            dset = getattr(torchvision.datasets, 'CIFAR100')
            dset_test = dset(self.data_dir, train=False, download=True)
            dset = dset(self.data_dir, train=True, download=True)
            data, targets = np.concatenate([dset.data, dset_test.data], axis=0), (dset.targets + dset_test.targets)
            new_ = targets
            for idx, target in enumerate(targets):
                new_[idx] = _cifar100_to_cifar20(target)
            targets = new_
        elif self.name == 'stl10':
            if self.train:
                if self.auxiliary:
                    split = 'train+test+unlabeled'
                else:
                    split = 'train+test'
            else:
                split = 'train+test'
            from .stl import STL10
            dset = STL10(root=self.data_dir, split=split, download=True)
            data = dset.data.transpose([0, 2, 3, 1])
            targets = dset.labels.astype(int)
        if self.train and not self.auxiliary:
            targets = np.load(self.paths['pretrained_target_path']).astype(int)
            assert (len(targets) == len(data))
        return data, targets

    def get_dset(self, use_strong_transform=False,
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        return BasicDataset(data, targets, num_classes, transform,
                            use_strong_transform, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     use_strong_transform=True, strong_transform=None,
                     onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair. 
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        if self.train:
            index = np.load(self.paths['clean_ind_path'])
            #             index = np.arange(num_labels)
            assert (num_labels == len(index))

        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets, num_labels, num_classes, index,
                                                                    include_lb_to_ulb)

        lb_dset = BasicDataset(lb_data, lb_targets, num_classes,
                               transform, False, None, onehot)

        ulb_dset = BasicDataset(ulb_data, ulb_targets, num_classes,
                                transform, use_strong_transform, strong_transform, onehot, cut_size[self.name])

        return lb_dset, ulb_dset


def _cifar100_to_cifar20(target):
    _dict = \
        {0: 4,
         1: 1,
         2: 14,
         3: 8,
         4: 0,
         5: 6,
         6: 7,
         7: 7,
         8: 18,
         9: 3,
         10: 3,
         11: 14,
         12: 9,
         13: 18,
         14: 7,
         15: 11,
         16: 3,
         17: 9,
         18: 7,
         19: 11,
         20: 6,
         21: 11,
         22: 5,
         23: 10,
         24: 7,
         25: 6,
         26: 13,
         27: 15,
         28: 3,
         29: 15,
         30: 0,
         31: 11,
         32: 1,
         33: 10,
         34: 12,
         35: 14,
         36: 16,
         37: 9,
         38: 11,
         39: 5,
         40: 5,
         41: 19,
         42: 8,
         43: 8,
         44: 15,
         45: 13,
         46: 14,
         47: 17,
         48: 18,
         49: 10,
         50: 16,
         51: 4,
         52: 17,
         53: 4,
         54: 2,
         55: 0,
         56: 17,
         57: 4,
         58: 18,
         59: 17,
         60: 10,
         61: 3,
         62: 2,
         63: 12,
         64: 12,
         65: 16,
         66: 12,
         67: 1,
         68: 9,
         69: 19,
         70: 2,
         71: 10,
         72: 0,
         73: 1,
         74: 16,
         75: 12,
         76: 9,
         77: 13,
         78: 15,
         79: 13,
         80: 16,
         81: 19,
         82: 2,
         83: 4,
         84: 6,
         85: 19,
         86: 5,
         87: 5,
         88: 8,
         89: 19,
         90: 18,
         91: 1,
         92: 2,
         93: 15,
         94: 6,
         95: 0,
         96: 17,
         97: 8,
         98: 14,
         99: 13}

    return _dict[target]
