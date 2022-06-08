import math
import torch
import torch.nn as nn
import torch.nn.functional as F

momentum = 0.001


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=37, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        assert (depth == 37)
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor, 128 * widen_factor]
        n = 4
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, 2, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        # 4th block
        self.block4 = NetworkBlock(n, channels[3], channels[4], block, 2, drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[4], num_classes)
        self.fc_auxiliary = nn.Linear(channels[4], num_classes + 5)
        self.channels = channels[4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    # def forward(self, x, auxiliary=False):
    #     out = self.conv1(x)
    #     out = self.block1(out)
    #     out = self.block2(out)
    #     out = self.block3(out)
    #     out = self.block4(out)
    #     out = self.relu(self.bn1(out))
    #     out = F.avg_pool2d(out, 6)
    #     out = out.view(-1, self.channels)
    #     if auxiliary:
    #         return self.fc(out), self.self.fc_auxiliary(out)
    #     else:
    #         return self.fc(out)

    def forward(self, x, auxiliary=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        if auxiliary:
            return self.fc(out), self.self.fc_auxiliary(out)
        else:
            return self.fc(out)

    # def forward(self, x, ood_test=False):
    #     out = self.conv1(x)
    #     out = self.block1(out)
    #     out = self.block2(out)
    #     out = self.block3(out)
    #     out = self.block4(out)
    #     out = self.relu(self.bn1(out))
    #     out = F.avg_pool2d(out, 6)
    #     out = out.view(-1, self.channels)
    #     if ood_test:
    #         return self.fc(out), out
    #     else:
    #         return self.fc(out)


class build_WideResNet:
    def __init__(self, depth=37, widen_factor=2, bn_momentum=0.01, dropRate=0.0):
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate

    def build(self, num_classes):
        return WideResNet(depth=self.depth,
                          widen_factor=self.widen_factor,
                          drop_rate=self.dropRate,
                          num_classes=num_classes)
