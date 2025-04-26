import os, sys, pathlib, logging
import numpy as np
import torch

__all__ = ["Trainer"]

logger = logging.getLogger(__name__)


class Trainer:
  def __init__(
    self,
    model,
    loss_fn,
    optimizer,
    train_data_loader,
    validation_data_loader,
    test_data_loader=None,
    hooks=[],
    keep_n_checkpoints=20,
    checkpoint_interval=1,
    save_path="train_results",
    device=torch.device("cpu"),
  ):
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.train_data_loader = train_data_loader
    self.validation_data_loader = validation_data_loader
    self.test_data_loader = test_data_loader
    self.hooks = hooks
    self.keep_n_checkpoints = keep_n_checkpoints
    self.checkpoint_interval = checkpoint_interval
    self.save_path = save_path
    self.device = device

    test_true = []
    if self.test_data_loader is not None:
      for test_batch in self.test_data_loader:
        test_true.extend(test_batch["DeltaG"].tolist())

    self.share_port = {"test_true": test_true}

    pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
    self.best_model_path = os.path.join(self.save_path, "best_model.pth")
    self.checkpoint_path = os.path.join(self.save_path, "checkpoints")

    pathlib.Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_files = [file for file in os.listdir(self.checkpoint_path) if file.startswith("checkpoint")]
    if checkpoint_files:
      self.restore_checkpoint()
    else:
      self.epoch = 0
      self.best_snapshot = {"epoch": 0, "loss": float("inf")}
      self.store_checkpoint()

    self.stop_training = False

  def _check_is_parallel(self):
    return True if isinstance(self.model, torch.nn.DataParallel) else False

  def _load_model_state_dict(self, state_dict):
    if self._check_is_parallel():
      self.model.module.load_state_dict(state_dict)
    else:
      self.model.load_state_dict(state_dict)

  def optimizer_to_device(self):
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if torch.is_tensor(v):
          state[k] = v.to(self.device)

  @property
  def state_dict(self):
    state_dict = {
      "epoch": self.epoch,
      "best_snapshot": self.best_snapshot,
      "optimizer": self.optimizer.state_dict(),
      "hooks": [hook.state_dict for hook in self.hooks],
    }
    if self._check_is_parallel():
      state_dict["model"] = self.model.module.state_dict()
    else:
      state_dict["model"] = self.model.state_dict()
    return state_dict

  @state_dict.setter
  def state_dict(self, state_dict):
    self.epoch = state_dict["epoch"]
    self.best_snapshot = state_dict["best_snapshot"]
    self.optimizer.load_state_dict(state_dict["optimizer"])
    self._load_model_state_dict(state_dict["model"])

    for hook, state_dict in zip(self.hooks, state_dict["hooks"]):
      hook.state_dict = state_dict

  def store_checkpoint(self):
    checkpoint = os.path.join(self.checkpoint_path, "checkpoint-" + str(self.epoch) + ".pth.tar")
    torch.save(self.state_dict, checkpoint)

    checkpoint_files = [file for file in os.listdir(self.checkpoint_path) if file.endswith(".pth.tar")]
    if self.keep_n_checkpoints > 0 and len(checkpoint_files) > self.keep_n_checkpoints:
      checkpoint_epochs = [int(file.split(".")[0].split("-")[-1]) for file in checkpoint_files]
      sorted_indices = np.argsort(checkpoint_epochs)
      for i in sorted_indices[: -self.keep_n_checkpoints]:
        os.remove(os.path.join(self.checkpoint_path, checkpoint_files[i]))

  def restore_checkpoint(self, epoch=None):
    if epoch is None:
      epoch = max([
        int(file.split(".")[0].split("-")[-1])
        for file in os.listdir(self.checkpoint_path)
        if file.startswith("checkpoint")
      ])

    checkpoint = os.path.join(self.checkpoint_path, "checkpoint-" + str(epoch) + ".pth.tar")
    self.state_dict = torch.load(checkpoint, map_location=self.device)

  def train(self, num_epochs=sys.maxsize):
    self.model.to(self.device)
    self.loss_fn.to(self.device)
    self.optimizer_to_device()
    self.stop_training = False

    for hook in self.hooks:
      hook.on_task_begin(self)
      
    last_epoch = self.epoch
    for _ in range(last_epoch, num_epochs):
      self.epoch += 1

      for hook in self.hooks:
        hook.on_epoch_begin(self)

      self.model.train()

      for hook in self.hooks:
        hook.on_train_begin(self)

      for train_batch in self.train_data_loader:
        for hook in self.hooks:
          hook.on_train_batch_begin(self)

        train_batch = [{k: v.to(self.device) for k, v in entry.items()} for entry in train_batch]
        train_result = self.model(train_batch)
        if self.model.__name__ == "DeepRLI" and not self.model.use_multi_obj:
          train_loss = self.loss_fn(train_result, train_batch["DeltaG"])
          for hook in self.hooks:
            hook.on_train_batch_end(self, train_loss.tolist(), train_batch["DeltaG"].shape[0])
        elif self.model.__name__ == "ContrastiveNet" and self.model.use_multi_obj:
          train_loss = self.loss_fn(train_result, [train_batch[0]["DeltaG"], train_batch[-1]["DeltaG"]])
          for hook in self.hooks:
            hook.on_train_batch_end(self, train_loss.tolist(), train_batch[0]["DeltaG"].shape[0])
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

      for hook in self.hooks:
        hook.on_train_end(self)

      self.model.eval()

      for hook in self.hooks:
        hook.on_validation_begin(self)

      for validation_batch in self.validation_data_loader:
        for hook in self.hooks:
          hook.on_validation_batch_begin(self)

        validation_batch = [{k: v.to(self.device) for k, v in entry.items()} for entry in validation_batch]
        validation_result = self.model(validation_batch)
        if self.model.__name__ == "DeepRLI" and not self.model.use_multi_obj:
          validation_loss = self.loss_fn(validation_result, validation_batch["DeltaG"])
          for hook in self.hooks:
            hook.on_validation_batch_end(self, validation_loss.tolist(), validation_batch["DeltaG"].shape[0])
        elif self.model.__name__ == "ContrastiveNet" and self.model.use_multi_obj:
          validation_loss = self.loss_fn(validation_result, [validation_batch[0]["DeltaG"], validation_batch[-1]["DeltaG"]])
          for hook in self.hooks:
            hook.on_validation_batch_end(self, validation_loss.tolist(), validation_batch[0]["DeltaG"].shape[0])

      for hook in self.hooks:
        hook.on_validation_end(self)

      if self.best_snapshot["loss"] > self.share_port["validation_loss"]:
        self.best_snapshot["epoch"] = self.epoch
        self.best_snapshot["loss"] = self.share_port["validation_loss"]
        if self._check_is_parallel():
          self.best_snapshot["model.state_dict"] = self.model.module.state_dict()
          torch.save(self.model.module, self.best_model_path)
        else:
          self.best_snapshot["model.state_dict"] = self.model.state_dict()
          torch.save(self.model, self.best_model_path)

      if self.test_data_loader:
        for hook in self.hooks:
          hook.on_test_begin(self)

        for test_batch in self.test_data_loader:
          for hook in self.hooks:
            hook.on_test_batch_begin(self)

          test_batch = [{k: v.to(self.device) for k, v in entry.items()} for entry in test_batch]
          test_result = self.model(test_batch)
          if self.model.__name__ == "DeepRLI" and not self.model.use_multi_obj:
            test_loss = self.loss_fn(test_result, test_batch["DeltaG"])
            for hook in self.hooks:
              hook.on_test_batch_end(self, test_loss.tolist(), test_batch["DeltaG"].shape[0])
          elif self.model.__name__ == "ContrastiveNet" and self.model.use_multi_obj:
            test_loss = self.loss_fn(test_result, [test_batch[0]["DeltaG"], test_batch[-1]["DeltaG"]])
            for hook in self.hooks:
              hook.on_test_batch_end(self, test_loss.tolist(), test_batch[0]["DeltaG"].shape[0])

        for hook in self.hooks:
          hook.on_test_end(self)

      for hook in self.hooks:
        hook.on_epoch_end(self)

      if self.epoch % self.checkpoint_interval == 0:
        self.store_checkpoint()

      if self.stop_training:
        break

    for hook in self.hooks:
      hook.on_task_end(self)

    self.store_checkpoint()
