import os, sys, time, argparse, pathlib, logging, json, pickle, random, secrets
from distutils.util import strtobool
import numpy as np
import torch

from deeprli.datasets import AffinityDatasetForTrain, ContrastiveDatasetForTrain
from deeprli.model import DeepRLI, ContrastiveNet
from deeprli.nn import MSELoss, ContrastiveLoss
from deeprli.train import Trainer
from deeprli.train.hooks import BaseHook, LoggingHook, ReduceLROnPlateauHook
from deeprli.utils import set_deterministic


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train DeepRLI Model")
  parser.add_argument('--config', type=str, help="the path of a configuration file")
  parser.add_argument("--model-name", type=str, help="the name of the model to be trained (DeepRLI or ContrastiveNet)")
  parser.add_argument("--train-data-root", type=str, help="the root path of training data")
  parser.add_argument("--train-data-index", type=str, help="the index file of training data (relative to the root)")
  parser.add_argument("--train-data-files", type=str, help="the path of training data files (relative to the root)")
  parser.add_argument("--test-data-root", type=str, help="the root path of test data")
  parser.add_argument("--test-data-index", type=str, help="the index file of test data (relative to the root)")
  parser.add_argument("--test-data-files", type=str, help="the path of test data files (relative to the root)")
  parser.add_argument("--epoch", type=int, help="training epochs")
  parser.add_argument("--batch", type=int, help="batch size")
  parser.add_argument("--num-workers", type=int, help="the number of subprocesses to use for data loading")
  parser.add_argument("--initial-lr", type=float, help="initial learning rate")
  parser.add_argument("--lr-reduction-factor", type=float, help="learning rate reduction factor")
  parser.add_argument("--lr-reduction-patience", type=int, help="learning rate reduction patience")
  parser.add_argument("--min-lr", type=float, help="lower bound on the learning rate")
  parser.add_argument("--weight-decay", type=float, help="weight decay (L2 penalty) of Adam")
  parser.add_argument("--f-dropout-rate", type=float, help="MLP dropout rate")
  parser.add_argument("--g-dropout-rate", type=float, help="GNN dropout rate")
  parser.add_argument("--hidden-dim", type=int, help="graph layer hidden dimension")
  parser.add_argument("--num-attention-heads", type=int, help="number of attention heads")
  parser.add_argument("--use-layer-norm", type=lambda x: bool(strtobool(x)), help="use layer normalization or not")
  parser.add_argument("--use-batch-norm", type=lambda x: bool(strtobool(x)), help="use batch normalization or not")
  parser.add_argument("--use-residual", type=lambda x: bool(strtobool(x)), help="use residual connections or not")
  parser.add_argument("--use-envelope", type=lambda x: bool(strtobool(x)), help="use envelope constraint or not")
  parser.add_argument("--use-multi-obj", type=lambda x: bool(strtobool(x)), help="use multi-objective or not")
  parser.add_argument("--loss-fn", type=str, help="loss function (MSELoss or ContrastiveLoss)")
  parser.add_argument("--gpu-id", type=int, help="the id of gpu")
  parser.add_argument("--seed", type=int, help="the random seed used to initialize pseudorandom number generators")
  parser.add_argument("--enable-data-parallel", type=lambda x: bool(strtobool(x)), help="enable data parallel or not")
  parser.add_argument("--use-all-train-data", type=lambda x: bool(strtobool(x)), help="use all train data or not")
  parser.add_argument("--initial-state", type=str, help="a state_dict stores the initial state of the model")
  parser.add_argument("--save-path", type=str, help="the path to deposit models, logs, et al.")
  args = parser.parse_args()

  if args.config and os.path.isfile(args.config):
    with open(args.config) as f:
      config = json.load(f)
  else:
    config = {}

  config_keys = [
    "model_name",
    "train_data_root",
    "train_data_index",
    "train_data_files",
    "test_data_root",
    "test_data_index",
    "test_data_files",
    "epoch",
    "batch",
    "num_workers",
    "initial_lr",
    "lr_reduction_factor",
    "lr_reduction_patience",
    "min_lr",
    "weight_decay",
    "f_dropout_rate",
    "g_dropout_rate",
    "hidden_dim",
    "num_attention_heads",
    "use_layer_norm",
    "use_batch_norm",
    "use_residual",
    "use_envelope",
    "use_multi_obj",
    "loss_fn",
    "gpu_id",
    "seed",
    "enable_data_parallel",
    "use_all_train_data",
    "initial_state",
    "save_path",
  ]
  
  config.update({key: value for key in config_keys if (value:=getattr(args, key, None)) is not None})
  if config.get("model_name") is None: config["model_name"] = "ContrastiveNet"
  if config.get("num_workers") is None: config["num_workers"] = 32
  if config.get("use_envelope") is None: config["use_envelope"] = True
  if config.get("use_multi_obj") is None: config["use_multi_obj"] = True
  if config.get("loss_fn") is None: config["loss_fn"] = "ContrastiveLoss"
  if config.get("seed") is None: config["seed"] = secrets.randbelow(1_000_000_000)
  if config.get("enable_data_parallel") is None: config["enable_data_parallel"] = False
  if config.get("use_all_train_data") is None: config["use_all_train_data"] = False

  pathlib.Path(config["save_path"]).mkdir(parents=True, exist_ok=True)

  logging.root.handlers = []
  logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
      logging.FileHandler(os.path.join(config["save_path"], "training.log"), 'w'),
      logging.StreamHandler(sys.stdout)
    ]
  )

  logger = logging.getLogger(__name__)
  logger.info("DeepRLI Training Records")

  now = time.localtime()
  timestamp = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
  logger.info(f"Date: {timestamp}")
  logger.info(config)

  device = torch.device(f"cuda:{config['gpu_id']}" if config.get("gpu_id") is not None and torch.cuda.is_available() else "cpu")

  logger.info("")
  logger.info(f"Device: " + ("GPU" if config.get("gpu_id") is not None and torch.cuda.is_available() else "CPU"))
  logger.info("=" * 80)
  if config["enable_data_parallel"]:
    logger.info(f"Training task on {device}, using {torch.cuda.device_count()} GPUs!")
  else:
    logger.info(f"Training task on {device}")

  logger.info("")
  logger.info(f"Random Seed")
  logger.info("=" * 80)
  logger.info(f"Value: {config['seed']}")
  set_deterministic(config["seed"])

  # load data
  if config["model_name"] == "DeepRLI" and not config["use_multi_obj"]:
    train_validation_data = AffinityDatasetForTrain(root=config["train_data_root"], data_index=config["train_data_index"], data_files=config["train_data_files"].split(','))
    if config.get("test_data_root") is not None:
      test_data = AffinityDatasetForTrain(root=config["test_data_root"], data_index=config["test_data_index"], data_files=config["test_data_files"].split(','))
  elif config["model_name"] == "ContrastiveNet":
    train_validation_data = ContrastiveDatasetForTrain(root=config["train_data_root"], data_index=config["train_data_index"], data_files=config["train_data_files"].split(','))
    if config.get("test_data_root") is not None:
      test_data = ContrastiveDatasetForTrain(root=config["test_data_root"], data_index=config["test_data_index"], data_files=config["test_data_files"].split(','))

  data_split_scheme_file = os.path.join(config["save_path"], "data_split_scheme.pkl")

  if os.path.isfile(data_split_scheme_file):
    logger.info("")
    logger.info(f"Use the existing data split scheme")
    with open(data_split_scheme_file, "rb") as f:
      data_split_scheme = pickle.load(f)
    train_data = torch.utils.data.dataset.Subset(train_validation_data, data_split_scheme["train_data_indices"])
    validation_data = torch.utils.data.dataset.Subset(train_validation_data, data_split_scheme["validation_data_indices"])
  else:
    train_size = int(0.9 * len(train_validation_data))
    validation_size = len(train_validation_data) - train_size
    train_data, validation_data = torch.utils.data.random_split(train_validation_data, [train_size, validation_size])
    with open(data_split_scheme_file, "wb") as f:
      pickle.dump({"train_data_indices": train_data.indices, "validation_data_indices": validation_data.indices}, f)

  if config["use_all_train_data"]:
    train_data = train_validation_data

  logger.info("")
  logger.info(f"Dataset Info")
  logger.info("=" * 80)
  logger.info(f"Dataset for train and validation: {train_validation_data}")
  logger.info("Data splitting ...")

  if config["model_name"] == "DeepRLI" and not config["use_multi_obj"]:
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch"], collate_fn=AffinityDatasetForTrain.collate_fn, shuffle=True, num_workers=config["num_workers"])
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=config["batch"], collate_fn=AffinityDatasetForTrain.collate_fn, num_workers=config["num_workers"])
    if config.get("test_data_root") is not None:
      test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch"], collate_fn=AffinityDatasetForTrain.collate_fn, num_workers=config["num_workers"])
    else:
      test_data_loader = None
  elif config["model_name"] == "ContrastiveNet":
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch"], collate_fn=ContrastiveDatasetForTrain.collate_fn, shuffle=True, num_workers=config["num_workers"])
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=config["batch"], collate_fn=ContrastiveDatasetForTrain.collate_fn, num_workers=config["num_workers"])
    if config.get("test_data_root") is not None:
      test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch"], collate_fn=ContrastiveDatasetForTrain.collate_fn, num_workers=config["num_workers"])
    else:
      test_data_loader = None

  logger.info(f"> Train Size: {len(train_data)}, Valid Size: {len(validation_data)}")
  if config.get("test_data_root") is not None:
    logger.info(f"> Test Size: {len(test_data)}")

  # build model
  if config["model_name"] == "DeepRLI":
    model = DeepRLI(
      f_dropout_rate=config["f_dropout_rate"],
      g_dropout_rate=config["g_dropout_rate"],
      hidden_dim=config["hidden_dim"],
      num_attention_heads=config["num_attention_heads"],
      use_layer_norm=config["use_layer_norm"],
      use_batch_norm=config["use_batch_norm"],
      use_residual=config["use_residual"],
      use_envelope=config["use_envelope"],
      use_multi_obj=config["use_multi_obj"]
    )
  elif config["model_name"] == "ContrastiveNet":
    model = ContrastiveNet(
      f_dropout_rate=config["f_dropout_rate"],
      g_dropout_rate=config["g_dropout_rate"],
      hidden_dim=config["hidden_dim"],
      num_attention_heads=config["num_attention_heads"],
      use_layer_norm=config["use_layer_norm"],
      use_batch_norm=config["use_batch_norm"],
      use_residual=config["use_residual"],
      use_envelope=config["use_envelope"],
      use_multi_obj=config["use_multi_obj"]
    )

  logger.info("")
  logger.info(f"Model: {model.__name__}")
  logger.info("=" * 80)
  logger.info(model)
  logger.info("Number of parameters: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))

  if config["enable_data_parallel"] and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    logger.info(f"Data parallel is enabled. Let's use {torch.cuda.device_count()} GPUs!")

  if config.get("initial_state") is not None:
    if config["enable_data_parallel"]:
      model.module.load_state_dict(torch.load(config["initial_state"], map_location=device))
    else:
      model.load_state_dict(torch.load(config["initial_state"], map_location=device))

  logger.info("")
  logger.info(f"Loss function & Optimizer")
  logger.info("=" * 80)
  logger.info(f'Loss function: {config["loss_fn"]}')

  if config["loss_fn"] == "MSELoss":
    loss_fn = MSELoss()
  elif config["loss_fn"] == "ContrastiveLoss":
    loss_fn = ContrastiveLoss()
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config["initial_lr"], weight_decay=config["weight_decay"])
  logger.info(f"Optimizer: {type(optimizer).__name__}")

  # add hooks
  hooks = [
    BaseHook(),
    LoggingHook(config["save_path"]),
    ReduceLROnPlateauHook(
      optimizer,
      factor=config["lr_reduction_factor"],
      patience=config["lr_reduction_patience"],
      min_lr=config["min_lr"],
      stop_after_min=True
    )
  ]

  # setup trainer
  trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader,
    test_data_loader=test_data_loader,
    hooks=hooks,
    save_path=config["save_path"],
    device=device
  )

  logger.info("")
  logger.info(f"Training: Epoch({config['epoch']}), Batch({config['batch']}) ...")
  logger.info('=' * 80)

  trainer.train(num_epochs=config["epoch"])
