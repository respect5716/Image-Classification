"""
DenseNet
Gao Huang, Densely Connected Convolutional Networks
https://arxiv.org/abs/1608.06993
"""
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_C, growth_rate):
        super(Bottleneck, self).__init__()
        C = growth_rate * self.expansion
        self.bn1 = nn.BatchNorm2d(in_C)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_C, C, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(C)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(C, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = torch.cat([x, shortcut], dim=1)
        return x


class Transition(nn.Module):
    def __init__(self, in_C, out_C):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_C)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_C, out_C, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


class DenseNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = cfg['growth_rate']
        self.in_C = self.growth_rate * 2
        self.head = nn.Sequential(
            nn.Conv2d(3, self.in_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_C),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(cfg['num_blocks'][0], True)
        self.layer2 = self._make_layer(cfg['num_blocks'][1], True)
        self.layer3 = self._make_layer(cfg['num_blocks'][2], True)
        self.layer4 = self._make_layer(cfg['num_blocks'][3], False)

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(self.in_C),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_C, num_classes)
        )


    def _make_layer(self, num_blocks, down_sample):
        layers = []
        for _ in range(num_blocks):
            layers.append(Bottleneck(self.in_C, self.growth_rate))
            self.in_C += self.growth_rate

        if down_sample:
            out_C = self.in_C // 2
            layers.append(Transition(self.in_C, out_C))
            self.in_C = out_C
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


def DenseNet121():
    cfg = {
        'growth_rate': 12,
        'num_blocks': [6, 12, 24, 16]
    }
    return DenseNet(cfg)