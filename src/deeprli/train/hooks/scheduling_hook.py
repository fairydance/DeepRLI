import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from deeprli.train.hooks import Hook

__all__ = ["ReduceLROnPlateauHook", "StepLRHook"]

class ReduceLROnPlateauHook(Hook):
  def __init__(self, optimizer, factor=0.5, patience=25, min_lr=1e-6, window_length=1, stop_after_min=False):
    self.scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr, verbose=True)
    self.window_length = window_length
    self.stop_after_min = stop_after_min
    self.window = []

  @property
  def state_dict(self):
    return {"scheduler": self.scheduler.state_dict()}

  @state_dict.setter
  def state_dict(self, state_dict):
    self.scheduler.load_state_dict(state_dict["scheduler"])

  def on_epoch_end(self, trainer):
    self.window.append(trainer.share_port["validation_loss"])
    if len(self.window) > self.window_length:
      self.window.pop(0)
    validation_loss = np.mean(self.window)

    self.scheduler.step(validation_loss)

    if self.stop_after_min:
      for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
        old_lr = float(param_group["lr"])
        if old_lr <= self.scheduler.min_lrs[i]:
          trainer.stop_training = True

class StepLRHook(Hook):
  def __init__(self, optimizer, step_size=30, gamma=0.1, min_lr=1e-6, stop_after_min=False):
    self.min_lr = min_lr
    self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    self.stop_after_min = stop_after_min

  @property
  def state_dict(self):
    return {"scheduler": self.scheduler.state_dict()}

  @state_dict.setter
  def state_dict(self, state_dict):
    self.scheduler.load_state_dict(state_dict["scheduler"])

  def on_epoch_end(self, trainer):
    self.scheduler.step()

    if self.stop_after_min and self.scheduler.get_last_lr()[0] < self.min_lr:
      trainer.stop_training = True