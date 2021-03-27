"""
ShuffleNet
Xiangyu Zhang, ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
https://arxiv.org/abs/1707.01083
"""
import torch
import torch.nn as nn

class Shuffle(nn.Module):
    def __init__(self, g):
        super(Shuffle, self).__init__()
        self.g = g
    
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, self.g, C//self.g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(B, C, H, W)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_C, out_C, groups, stride):
        super(Bottleneck, self).__init__()
        C = out_C // self.expansion
        self.stride = stride
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        
        self.conv1 = nn.Conv2d(in_C, C, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.relu1 = nn.ReLU()
        self.shuffle = Shuffle(groups)
        
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(C, out_C, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_C)
        self.relu3 = nn.ReLU()

    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.shuffle(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.stride > 1:
            shortcut = self.pool(x)
            out = torch.cat([out, shortcut], dim=1)
        else:
            out += x
        out = self.relu3(out)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(ShuffleNet, self).__init__()
        self.groups = cfg['groups']
        self.in_C = cfg['in_C']
        self.head = nn.Sequential(
            nn.Conv2d(3, self.in_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_C),
            nn.ReLU()
        )
    
        self.layer1 = self._make_layer(cfg['out_C'][0], cfg['num_blocks'][0])
        self.layer2 = self._make_layer(cfg['out_C'][1], cfg['num_blocks'][1])
        self.layer3 = self._make_layer(cfg['out_C'][2], cfg['num_blocks'][2])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_C, num_classes)
        )

    def _make_layer(self, out_C, num_blocks):
        layers = []
        for i in range(num_blocks):
            st = 1 if i else 2
            modified_out_C = out_C if i else out_C - self.in_C
            layers.append(Bottleneck(self.in_C, modified_out_C, self.groups, st))
            self.in_C = out_C
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x

def ShuffleNet32():
    cfg = {
        'groups': 2,
        'in_C': 24,
        'out_C': [200, 400, 800],
        'num_blocks': [4, 8, 4]
    }
    return ShuffleNet(cfg)