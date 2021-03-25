"""
ResNet
Kaiming He, Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
"""

import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_C, C, stride):
        super(Bottleneck, self).__init__()
        out_C = C * self.expansion
        self.shortcut = nn.Sequential()
        if stride > 1 or in_C != out_C:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_C, out_C, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_C)
            )

        self.conv1 = nn.Conv2d(in_C, C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(C, out_C, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_C)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += res
        x = self.relu3(x)
        return x


class ResNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(ResNet, self).__init__()
        self.in_C = cfg['in_C']
        self.head = nn.Sequential(
            nn.Conv2d(3, self.in_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_C),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(cfg['C'][0], cfg['num_blocks'][0], 1)
        self.layer2 = self._make_layer(cfg['C'][1], cfg['num_blocks'][1], 2)
        self.layer3 = self._make_layer(cfg['C'][2], cfg['num_blocks'][2], 2)
        self.layer4 = self._make_layer(cfg['C'][3], cfg['num_blocks'][3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_C, num_classes)
    
    def _make_layer(self, C, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(Bottleneck(self.in_C, C, st))
            self.in_C = C * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view((-1, self.in_C))
        x = self.classifier(x)
        return x


def ResNet32():
    cfg = {
        'in_C': 32,
        'C': [32, 32, 64, 128],
        'num_blocks': [2, 3, 3, 2]
    }
    return ResNet(cfg)