"""
VGG (with Batch normalization)
Karen Simonyan, Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556v6
"""

import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_C, out_C):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_C, out_C, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_C)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layer(cfg)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_C, num_classes)
        )
        
    def _make_layer(self, cfg):
        layers = []
        self.in_C = 3

        for c in cfg:
            if c == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(Block(self.in_C, c))
                self.in_C = c
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def VGG11():
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(cfg)