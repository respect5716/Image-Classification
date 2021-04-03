
"""
DPN
Yunpeng Chen, Dual Path Network
https://arxiv.org/abs/1707.01629
"""
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_C, C, residual_C, dense_C, stride, is_first):
        super(Bottleneck, self).__init__()
        self.c = residual_C
        out_C = residual_C + dense_C
        self.shortcut = nn.Sequential()
        if is_first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_C, out_C, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_C)
            )
        
        self.conv1 = nn.Conv2d(in_C, C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, groups=24, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(C, out_C, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_C)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.cat([x[:,:self.c,:,:] + shortcut[:,:self.c,:,:], x[:,self.c:,:,:], shortcut[:,self.c:,:,:]], dim=1)
        x = self.relu3(x)
        return x


class DPN(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(DPN, self).__init__()
        self.in_C = cfg['in_C']
        self.head = nn.Sequential(
            nn.Conv2d(3, self.in_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_C)
        )

        self.layer1 = self._make_layer(cfg['C'][0], cfg['residual_C'][0], cfg['dense_C'][0], cfg['num_blocks'][0], 1)
        self.layer2 = self._make_layer(cfg['C'][1], cfg['residual_C'][1], cfg['dense_C'][1], cfg['num_blocks'][1], 2)
        self.layer3 = self._make_layer(cfg['C'][2], cfg['residual_C'][2], cfg['dense_C'][2], cfg['num_blocks'][2], 2)
        self.layer4 = self._make_layer(cfg['C'][3], cfg['residual_C'][3], cfg['dense_C'][3], cfg['num_blocks'][3], 2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_C, num_classes)
        )

    def _make_layer(self, C, residual_C, dense_C, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, st in enumerate(strides):
            layers.append(Bottleneck(self.in_C, C, residual_C, dense_C, st, i==0))
            self.in_C = residual_C + (i+2) * dense_C
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


def DPN26():
    cfg = {
        'in_C': 64,
        'C': [48, 96, 192, 384],
        'residual_C': [64, 128, 256, 512],
        'dense_C': [16, 32, 64, 128],
        'num_blocks': [2, 2, 2, 2]
    }
    return DPN(cfg)