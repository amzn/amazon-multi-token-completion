from torch.optim import Optimizer
from lr_scheduler.lr_scheduler import LearningRateScheduler
from lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Warmup learning rate until `warmup_steps` and reduce learning rate on plateau after.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            warmup_steps: int,
            patience: int = 5,
            factor: float = 0.2,
    ) -> None:
        super(WarmupReduceLROnPlateauScheduler, self).__init__(optimizer, init_lr)
        self.warmup_steps = warmup_steps
        self.update_steps = 0
        self.warmup_rate = (peak_lr - init_lr) / self.warmup_steps \
            if self.warmup_steps != 0 else 0
        self.schedulers = [
            WarmupLRScheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps,
            ),
            ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor)
        ]

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0
        else:
            return 1

    def step(self, metrics, epoch=None):
        stage = self._decide_stage()
        if stage == 0:
            self.schedulers[0].step()
        elif stage == 1:
            self.schedulers[1].step(metrics=metrics)
        self.update_steps += 1
        return self.get_lr()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
