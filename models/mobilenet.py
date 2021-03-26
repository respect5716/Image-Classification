"""
MobileNet
Andrew G. Howard, MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861
"""

import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_C, out_C, stride):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_C, in_C, kernel_size=3, stride=stride, padding=1, groups=in_C, bias=False)
        self.bn1 = nn.BatchNorm2d(in_C)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_C, out_C, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_C)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class MobileNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(MobileNet, self).__init__()
        self.in_C = cfg['C'][0]
        self.head = nn.Sequential(
            nn.Conv2d(3, self.in_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_C),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(cfg['C'][0], cfg['num_blocks'][0], 1)
        self.layer2 = self._make_layer(cfg['C'][1], cfg['num_blocks'][1], 2)
        self.layer3 = self._make_layer(cfg['C'][2], cfg['num_blocks'][2], 2)
        self.layer4 = self._make_layer(cfg['C'][3], cfg['num_blocks'][3], 2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_C, num_classes)
        )

    def _make_layer(self, C, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(Block(self.in_C, C, st))
            self.in_C = C
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


def MobileNet22():
    cfg = {
        'C': [32, 32, 64, 128],
        'num_blocks': [2, 3, 3, 2]
    }
    return MobileNet(cfg)