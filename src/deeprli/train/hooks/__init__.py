class Hook:
  """Base class for hooks."""

  @property
  def state_dict(self):
    return {}

  @state_dict.setter
  def state_dict(self, state_dict):
    pass

  def on_task_begin(self, trainer):
    pass

  def on_epoch_begin(self, trainer):
    pass

  def on_train_begin(self, trainer):
    pass

  def on_train_batch_begin(self, trainer):
    pass

  def on_train_batch_end(self, trainer, train_loss, train_size):
    pass

  def on_train_end(self, trainer):
    pass

  def on_validation_begin(self, trainer):
    pass

  def on_validation_batch_begin(self, trainer):
    pass

  def on_validation_batch_end(self, trainer, validation_loss, validation_size):
    pass

  def on_validation_end(self, trainer):
    pass

  def on_test_begin(self, trainer):
    pass

  def on_test_batch_begin(self, trainer):
    pass

  def on_test_batch_end(self, trainer, test_loss, test_size, test_result):
    pass

  def on_test_end(self, trainer):
    pass

  def on_epoch_end(self, trainer):
    pass

  def on_task_end(self, trainer):
    pass

from .base_hook import *
from .logging_hook import *
from .scheduling_hook import *