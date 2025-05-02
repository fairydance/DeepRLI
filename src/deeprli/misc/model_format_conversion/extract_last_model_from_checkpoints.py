import os, argparse, json
from collections import OrderedDict
from distutils.util import strtobool
import torch

from deeprli.model import DeepRLI


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Get last model from checkpoints")
  parser.add_argument('--config', type=str, help="the path of a configuration file")
  parser.add_argument("--checkpoint-path", type=str, help="the path deposits checkpoints")
  parser.add_argument("--remove-prefix", type=lambda x: bool(strtobool(x)), help="remove prefix")
  parser.add_argument("--save-type", type=str, help="state_dict or model_object")
  parser.add_argument("--save-path", type=str, help="the path to deposit state_dict")
  parser.add_argument("--f-dropout-rate", type=float, help="input feature dropout")
  parser.add_argument("--g-dropout-rate", type=float, help="graph feature dropout")
  parser.add_argument("--hidden-dim", type=int, help="graph layer hidden dimension")
  parser.add_argument("--num-attention-heads", type=int, help="number of attention heads")
  parser.add_argument("--use-layer-norm", type=lambda x: bool(strtobool(x)), help="use layer normalization or not")
  parser.add_argument("--use-batch-norm", type=lambda x: bool(strtobool(x)), help="use batch normalization or not")
  parser.add_argument("--use-residual", type=lambda x: bool(strtobool(x)), help="use residual connections or not")
  parser.add_argument("--use-envelope", type=lambda x: bool(strtobool(x)), help="use envelope constraint or not")
  parser.add_argument("--use-multi-obj", type=lambda x: bool(strtobool(x)), help="use multi-objective or not")
  args = parser.parse_args()

  if args.config is not None and os.path.isfile(args.config):
    with open(args.config) as f:
      config = json.load(f)
  else:
    config = {}

  config_keys = [
    "checkpoint_path",
    "remove_prefix",
    "save_type",
    "save_path",
    "f_dropout_rate",
    "g_dropout_rate",
    "hidden_dim",
    "num_attention_heads",
    "use_layer_norm",
    "use_batch_norm",
    "use_residual",
    "use_envelope",
    "use_multi_obj",
  ]

  config.update({key: value for key in config_keys if (value:=getattr(args, key, None)) is not None})
  if config.get("remove_prefix") is None: config["remove_prefix"] = False
  if config.get("use_envelope") is None: config["use_envelope"] = True
  if config.get("use_multi_obj") is None: config["use_multi_obj"] = True
  print(config)

  device = torch.device("cpu")

  checkpoint_files = [file for file in os.listdir(config["checkpoint_path"]) if file.startswith("checkpoint")]
  if checkpoint_files:
    epoch = max([int(file.split(".")[0].split("-")[-1]) for file in checkpoint_files])
    checkpoint = os.path.join(config["checkpoint_path"], "checkpoint-" + str(epoch) + ".pth.tar")
    print(f"Get model from {checkpoint}")
    state_snapshot = torch.load(checkpoint, map_location=device)
    if config["remove_prefix"]:
      state_snapshot["model"] = OrderedDict((k.removeprefix("potential."), v) for k, v in state_snapshot["model"].items())
    if config["save_type"] == "state_dict":
      torch.save(state_snapshot["model"], config["save_path"])
    elif config["save_type"] == "model_object":
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
      model.load_state_dict(state_snapshot["model"])
      torch.save(model, config["save_path"])
    else:
      raise Exception("Unkown save type.")
  else:
    raise Exception("No checkpoint file is found in the given path.")
