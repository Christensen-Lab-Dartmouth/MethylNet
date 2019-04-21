"""
schedulers.py
=======================
Learning rate schedulers that help enable better and more generalizable models."""

import torch
from torch.optim.lr_scheduler import ExponentialLR,LambdaLR
import math


class CosineAnnealingWithRestartsLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Borrowed from: https://github.com/mpyrozhok/adamwr/blob/master/cyclic_scheduler.py
    Needs to be updated to reflect newest changes.
    From original docstring:
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
     .. math::
         \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
     When last_epoch=-1, sets initial lr as lr.
     It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implements
    the cosine annealing part of SGDR, the restarts and number of iterations multiplier.
     Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Multiply T_max by this number after each restart. Default: 1.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
     .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1., alpha_decay=1.0):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        self.alpha = alpha_decay
        super().__init__(optimizer, last_epoch)

    def restart(self):
        self.restarts += 1
        self.restart_every = int(round(self.restart_every * self.T_mult))
        self.restarted_at = self.last_epoch

    def cosine(self, base_lr):
        return self.eta_min + self.alpha**self.restarts * (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2

    @property
    def step_n(self):
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        if self.step_n >= self.restart_every:
            self.restart()
        return [self.cosine(base_lr) for base_lr in self.base_lrs]

class Scheduler:
    """Scheduler class that modulates learning rate of torch optimizers over epochs.

    Parameters
    ----------
    optimizer : type
        torch.Optimizer object
    opts : type
        Options of setting the learning rate scheduler, see default.

    Attributes
    ----------
    schedulers : type
        Different types of schedulers to choose from.
    scheduler_step_fn : type
        How scheduler updates learning rate.
    initial_lr : type
        Initial set learning rate.
    scheduler_choice : type
        What scheduler type was chosen.
    scheduler : type
        Scheduler object chosen that will more directly update optimizer LR.

    """

    def __init__(self, optimizer=None, opts=dict(scheduler='null',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2)):
        self.schedulers = {'exp':(lambda optimizer: ExponentialLR(optimizer, opts["lr_scheduler_decay"])),
                            'null':(lambda optimizer: None),
                            'warm_restarts':(lambda optimizer: CosineAnnealingWithRestartsLR(optimizer, T_max=opts['T_max'], eta_min=opts['eta_min'], last_epoch=-1, T_mult=opts['T_mult']))}
        self.scheduler_step_fn = {'exp':(lambda scheduler: scheduler.step()),
                                  'warm_restarts':(lambda scheduler: scheduler.step()),
                                  'null':(lambda scheduler: None)}
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.scheduler_choice = opts['scheduler']
        self.scheduler = self.schedulers[self.scheduler_choice](optimizer) if optimizer is not None else None

    def step(self):
        """Update optimizer learning rate"""
        self.scheduler_step_fn[self.scheduler_choice](self.scheduler)

    def get_lr(self):
        """Return current learning rate.

        Returns
        -------
        float
            Current learning rate.

        """
        lr = (self.initial_lr if self.scheduler_choice == 'null' else self.scheduler.optimizer.param_groups[0]['lr'])
        return lr
