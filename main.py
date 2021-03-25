import os
import random
import wandb
import numpy as np

import torch
import torch.nn as nn

from data import create_dataloader
from models import create_model, Model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vgg')
parser.add_argument('--project', type=str, default='cifar10')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch_size', type=int, default=200)
args = parser.parse_args()
CONFIG = vars(args)

SEED = 23
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def prepare():
    train_loader, val_loader, test_loader = create_dataloader(CONFIG['batch_size'])
    _model = create_model(CONFIG['model'])
    optimizer = optimizer = torch.optim.SGD(_model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(_model, optimizer, scheduler, criterion, device)
    model.summary()
    return model, train_loader, val_loader, test_loader


def main():
    model, train_loader, val_loader, test_loader = prepare()
    model.fit(
        train_loader,
        val_loader,
        epoch_size = CONFIG['epoch_size']
    )
    model.evaluate(test_loader)

if __name__ == '__main__':
    main()