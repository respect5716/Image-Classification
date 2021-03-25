import time
import wandb

import torch
import torchsummary

from .vgg import *
from .resnet import ResNet50


def create_model(model_name):
    model_dict = {
        'vgg': VGG11,
        'resnet': ResNet50
    }

    _model = model_dict[model_name]()
    return _model


class Model(object):
    def __init__(self, _model, optimizer, scheduler, criterion, device):
        self._model = _model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        wandb.watch(self._model)
        self.best_epoch, self.best_score = 0, 0.0
    

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        outputs = self._model(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        
        _, pred = outputs.max(1)
        acc = pred.eq(y).sum().item() / pred.size(0)
        loss = loss.item()
        return loss, acc
    

    def test_step(self, x, y):
        with torch.no_grad():
            outputs = self._model(x)
            loss = self.criterion(outputs, y)
            _, pred = outputs.max(1)
            acc = pred.eq(y).sum().item() / pred.size(0)
            loss = loss.item()
        return loss, acc
    

    def run_epoch(self, loader, phase):
        if phase == 'train':
            self._model.train()
            step_fn = self.train_step
        else:
            self._model.eval()
            step_fn = self.test_step

        loss, acc = 0.0, 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _loss, _acc = step_fn(x, y)
            loss += _loss
            acc += _acc
        
        loss /= len(loader)
        acc /= len(loader)
        return loss, acc
    

    def fit(self, train_loader, val_loader, epoch_size):
        print("=" * 65)
        print(f"Train started!! (device: {self.device})")

        for ep in range(epoch_size):
            ep_start = time.time()
            train_loss, train_acc = self.run_epoch(train_loader, 'train')
            val_loss, val_acc = self.run_epoch(val_loader, 'test')
            ep_end = time.time()
            ep_time = ep_end - ep_start

            if val_acc > self.best_score:
                torch.save(self._model.state_dict(), "model.pth")
                self.best_epoch = ep
                self.best_score = val_acc
            self.scheduler.step()
            wandb.log({"epoch":ep, "train_loss":train_loss, "train_acc":train_acc, "val_loss":val_loss, "val_acc":val_acc, "best_epoch":self.best_epoch, "best_score":self.best_score})
            print(f"EP {ep:03d} | Train Loss {train_loss:.3f} | Train Acc {train_acc:.3f} | Val Loss {val_loss:.3f} | Val Acc {val_acc:.3f} | Time {ep_time:.0f}s")

        print(f"Train Finished!!")
        print("=" * 65)

    def evaluate(self, test_loader):
        ep_start = time.time()
        self._model.load_state_dict(torch.load("model.pth", map_location=self.device))
        test_loss, test_acc = self.run_epoch(test_loader, 'test')
        ep_end = time.time()
        ep_time = ep_end - ep_start
        wandb.log({"test_loss":test_loss, "test_acc":test_acc})
        print(f"Eval | Test Loss {test_loss:.3f} | Test Acc {test_acc:.3f} | Time {ep_time:.0f}s")
    
    
    def summary(self):
        print(torchsummary.summary(self._model, input_size=(3, 32, 32), device=self.device))