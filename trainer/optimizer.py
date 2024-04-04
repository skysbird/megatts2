import numpy as np
import torch

class ScheduledOptim(torch.optim.lr_scheduler._LRScheduler):
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_steps):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr_frozen(self, learning_rate_frozen):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate_frozen
        self.optimizer.step()

    def step(self):
        self._update_learning_rate()
        self.optimizer.step()

    def get_learning_rate(self):
        learning_rate = 0.0
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def zero_grad(self):
        # print(self.init_lr)
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
