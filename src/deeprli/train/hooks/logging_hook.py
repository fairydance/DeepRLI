import os, logging
from deeprli.train.hooks import Hook

__all__ = ["LoggingHook"]

logger = logging.getLogger(__name__)

class LoggingHook(Hook):
  def __init__(self, save_path, log_interval=1):
    self.save_path = save_path
    self.log_interval = log_interval
    self.open_mode = 'w'

  @property
  def state_dict(self):
    return {"open_mode": self.open_mode}

  @state_dict.setter
  def state_dict(self, state_dict):
    self.open_mode = state_dict["open_mode"]

  def on_task_begin(self, trainer):
    self.train_log = open(os.path.join(self.save_path, "training.csv"), self.open_mode)
    if self.open_mode == 'w':
      if trainer.share_port["test_true"]:
        self.train_log.write("Epoch,Learning Rate,Train Loss,Validation Loss,Test Loss,Test Pearson,Time\n")
      else:
        self.train_log.write("Epoch,Learning Rate,Train Loss,Validation Loss,Time\n")
      self.open_mode = 'a'

  def on_epoch_end(self, trainer):
    if trainer.epoch % self.log_interval == 0:
      if trainer.share_port["test_true"]:
        self.train_log.write(f'{trainer.epoch},{trainer.share_port["learning_rate"]},{trainer.share_port["train_loss"]},{trainer.share_port["validation_loss"]},{trainer.share_port["test_loss"]},{trainer.share_port["test_pearson"]},{trainer.share_port["elapsed_time"]}\n')
        logger.info("| Epoch {:d} | Learning Rate {:.3e} | Train Loss {:.3f} | Validation Loss {:.3f} | Test Loss {:.3f} | Test Pearson {:.3f} | Time {:.3f} s |".format(
          trainer.epoch, trainer.share_port["learning_rate"], trainer.share_port["train_loss"], trainer.share_port["validation_loss"], trainer.share_port["test_loss"], trainer.share_port["test_pearson"], trainer.share_port["elapsed_time"]))
      else:
        self.train_log.write(f'{trainer.epoch},{trainer.share_port["learning_rate"]},{trainer.share_port["train_loss"]},{trainer.share_port["validation_loss"]},{trainer.share_port["elapsed_time"]}\n')
        logger.info("| Epoch {:d} | Learning Rate {:.3e} | Train Loss {:.3f} | Validation Loss {:.3f} | Time {:.3f} s |".format(
          trainer.epoch, trainer.share_port["learning_rate"], trainer.share_port["train_loss"], trainer.share_port["validation_loss"], trainer.share_port["elapsed_time"]))

  def on_task_end(self, trainer):
    self.train_log.close()
    logger.info(f'Mission complete! The best model is in epoch {trainer.best_snapshot["epoch"]} with loss {trainer.best_snapshot["loss"]}')