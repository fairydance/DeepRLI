import time
import numpy as np
import torch
from deeprli.train.hooks import Hook
from deeprli.utils.metrics import pearson

__all__ = ["BaseHook"]

class BaseHook(Hook):
  def on_epoch_begin(self, trainer):
    torch.cuda.synchronize()
    trainer.share_port["start_time"] = time.time()
    trainer.share_port["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]

  def on_train_begin(self, trainer):
    trainer.share_port["train_loss"] = []
    trainer.share_port["train_size"] = []

  def on_train_batch_end(self, trainer, train_loss, train_size):
    trainer.share_port["train_loss"].append(train_loss * train_size)
    trainer.share_port["train_size"].append(train_size)
    
  def on_train_end(self, trainer):
    trainer.share_port["train_loss"] = np.sum(trainer.share_port["train_loss"]) / np.sum(trainer.share_port["train_size"])

  def on_validation_begin(self, trainer):
    trainer.share_port["validation_loss"] = []
    trainer.share_port["validation_size"] = []

  def on_validation_batch_end(self, trainer, validation_loss, validation_size):
    trainer.share_port["validation_loss"].append(validation_loss * validation_size)
    trainer.share_port["validation_size"].append(validation_size)

  def on_validation_end(self, trainer):
    trainer.share_port["validation_loss"] = np.sum(trainer.share_port["validation_loss"]) / np.sum(trainer.share_port["validation_size"])

  def on_test_begin(self, trainer):
    trainer.share_port["test_loss"] = []
    trainer.share_port["test_size"] = []
    trainer.share_port["test_pred"] = []

  def on_test_batch_end(self, trainer, test_loss, test_size, test_result):
    trainer.share_port["test_loss"].append(test_loss * test_size)
    trainer.share_port["test_size"].append(test_size)
    trainer.share_port["test_pred"].extend(test_result.tolist())

  def on_test_end(self, trainer):
    trainer.share_port["test_loss"] = np.sum(trainer.share_port["test_loss"]) / np.sum(trainer.share_port["test_size"])
    trainer.share_port["test_pearson"] = pearson(trainer.share_port["test_pred"], trainer.share_port["test_true"])

  def on_epoch_end(self, trainer):
    torch.cuda.synchronize()
    trainer.share_port["elapsed_time"] = time.time() - trainer.share_port["start_time"]
