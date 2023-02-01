import torch
from typing import Optional
from torch.optim import Optimizer

from lr_scheduler.lr_scheduler import LearningRateScheduler


class WarmupLRScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            warmup_steps: int,
            constant=False
    ) -> None:
        super(WarmupLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.constant = constant
        if warmup_steps != 0:
            warmup_rate = peak_lr - init_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.lr = init_lr
        self.warmup_steps = warmup_steps

        self.set_lr(self.optimizer, self.init_lr)

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps < self.warmup_steps and not self.constant:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        elif self.update_steps == self.warmup_steps:
            self.set_lr(self.optimizer, self.peak_lr)
            self.lr = self.peak_lr
        self.update_steps += 1
        return self.lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
