import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, scheduler):
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        super(WarmupScheduler, self).__init__(optimizer, -1)
        self._last_lr = [0.0] * len(optimizer.param_groups)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        return self.scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            self.last_epoch += 1
            new_lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
                param_group['lr'] = lr
            self._last_lr = new_lrs
        else:
            self.scheduler.step(epoch)
            self._last_lr = self.scheduler.get_last_lr()