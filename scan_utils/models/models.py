"""
References: https://github.com/wvangansbeke/Unsupervised-Classification.git
(Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/))
"""
import torch.nn as nn
import torch.nn.functional as F

from .wrn import WideResNet


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim=1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            backbone = self.backbone(x)
            features = F.normalize(backbone, dim=1)
            out = {'features': features, 'output': [cluster_head(backbone) for cluster_head in self.cluster_head]}

        elif forward_pass == 'features':
            backbone = self.backbone(x)
            out = F.normalize(backbone, dim=1)

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out


class Wrn(nn.Module):
    def __init__(self, num_classes=10, filters=32, repeat=4):
        super(Wrn, self).__init__()
        self.backbone = WideResNet(num_classes=num_classes, filters=filters, repeat=repeat)

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            out = [self.backbone(x)]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'return_all':
            backbone = self.backbone(x)
            features = F.normalize(backbone, dim=1)
            out = {'features': features, 'output': [backbone]}

        elif forward_pass == 'features':
            backbone = self.backbone(x)
            out = F.normalize(backbone, dim=1)

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out
