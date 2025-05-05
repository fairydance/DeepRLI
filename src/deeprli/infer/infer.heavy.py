import os, sys, time, argparse, pathlib, logging, json
from distutils.util import strtobool
import pandas as pd
import torch

from deeprli.datasets import HeavyDatasetForInfer
from deeprli.model import DeepRLI, DeepRLIForInterpretation

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Infer DeepRLI (heavy)")
  parser.add_argument('--config', type=str, help="the path of a configuration file")
  parser.add_argument("--model", type=str, help="the path of a trained model or an ensemble of trained models")
  parser.add_argument("--model-format", type=str, help="state_dict or model_object")
  parser.add_argument("--interpretation", type=lambda x: bool(strtobool(x)), help="with interpretation or not")
  parser.add_argument("--data-root", type=str, help="the root path of data")
  parser.add_argument("--data-index", type=str, help="the path of an index file relative to the root")
  parser.add_argument("--data-file", type=str, help="the path of a data file relative to the root")
  parser.add_argument("--batch", type=int, help="batch size")
  parser.add_argument("--num-workers", type=int, help="the number of subprocesses to use for data loading")
  parser.add_argument("--gpu-id", type=int, help="the id of gpu")
  parser.add_argument("--save-path", type=str, help="the path to deposit results")
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

  if args.config and os.path.isfile(args.config):
    with open(args.config) as f:
      config = json.load(f)
  else:
    config = {}

  config_keys = [
    "model",
    "model_format",
    "interpretation",
    "data_root",
    "data_index",
    "data_file",
    "batch",
    "num_workers",
    "gpu_id",
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
  if config.get("model") is None: config["model"] = "trained_model.state_dict.pth"
  if config.get("model_format") is None: config["model_format"] = "state_dict"
  if config.get("interpretation") is None: config["interpretation"] = False
  if config.get("batch") is None: config["batch"] = 16
  if config.get("num_workers") is None: config["num_workers"] = 32
  if config.get("gpu_id") is None: config["gpu_id"] = 0
  if config.get("save_path") is None: config["save_path"] = ""
  if config.get("f_dropout_rate") is None: config["f_dropout_rate"] = 0.0
  if config.get("g_dropout_rate") is None: config["g_dropout_rate"] = 0.0
  if config.get("hidden_dim") is None: config["hidden_dim"] = 64
  if config.get("num_attention_heads") is None: config["num_attention_heads"] = 8
  if config.get("use_layer_norm") is None: config["use_layer_norm"] = False
  if config.get("use_batch_norm") is None: config["use_batch_norm"] = True
  if config.get("use_residual") is None: config["use_residual"] = True
  if config.get("use_envelope") is None: config["use_envelope"] = True
  if config.get("use_multi_obj") is None: config["use_multi_obj"] = True

  pathlib.Path(config["save_path"]).mkdir(parents=True, exist_ok=True)

  logging.root.handlers = []
  logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
      logging.FileHandler(os.path.join(config["save_path"], "inferring.log"), 'w'),
      logging.StreamHandler(sys.stdout)
    ]
  )

  logger = logging.getLogger(__name__)
  logger.info("DeepRLI Inferring Records")

  now = time.localtime()
  timestamp = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
  logger.info(f"Date: {timestamp}")
  logger.info(config)

  device = torch.device(f"cuda:{config['gpu_id']}" if config.get("gpu_id") is not None and torch.cuda.is_available() else "cpu")

  logger.info("")
  logger.info(f"Device: " + ("GPU" if config.get("gpu_id") is not None and torch.cuda.is_available() else "CPU"))
  logger.info("=" * 80)
  logger.info(f"Inferring task on {device}")

  # load data
  data = HeavyDatasetForInfer(root=config["data_root"], data_index=config["data_index"], data_file=config["data_file"])

  logger.info("")
  logger.info(f"Dataset Info")
  logger.info("=" * 80)
  logger.info(f"Dataset to be inferred: {data}")
  logger.info(f"> Size: {len(data)}")

  data_loader = torch.utils.data.DataLoader(data, batch_size=config["batch"], collate_fn=HeavyDatasetForInfer.collate_fn, num_workers=config["num_workers"])

  if not config["interpretation"]:
    if config["model_format"] == "model_object":
      model = torch.load(config["model"], map_location=device)
    elif config["model_format"] == "state_dict":
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
      model.to(device)
      model.load_state_dict(torch.load(config["model"], map_location=device))

    model.eval()
    p = [[], [], []]
    start = time.perf_counter()
    for batch in data_loader:
      batch = {k: v.to(device) for k, v in batch.items()}
      result = model(batch)
      for i in range(3):
        p[i].extend(result[i].tolist())
    end = time.perf_counter()
  else:
    pathlib.Path(os.path.join(config["save_path"], "results")).mkdir(parents=True, exist_ok=True)
    if config["model_format"] == "model_object":
      model = torch.load(config["model"], map_location=device)
    elif config["model_format"] == "state_dict":
      model = DeepRLIForInterpretation(
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
      model.to(device)
      model.load_state_dict(torch.load(config["model"], map_location=device))

    model.eval()
    p = [[], [], []]
    start = time.perf_counter()
    for j, batch in enumerate(data_loader):
      batch = {k: v.to(device) for k, v in batch.items()}
      result = model(batch)
      torch.save(result, os.path.join(config["save_path"], "results", f"result_{j}.pth"))
      for i in range(3):
        p[i].extend(result["scores"][i].tolist())
    end = time.perf_counter()

  duration = end - start
  avg_per_item = duration / len(data)

  logger.info(f"Infering complete. Time of duration: {duration:.6f} seconds for {len(data)} item(s). Average per item: {avg_per_item:.6f} seconds/item")

  inference_results = pd.read_csv(os.path.join(config["data_root"], config["data_index"]))
  inference_results["scoring_score"] = p[0]
  inference_results["docking_score"] = p[1]
  inference_results["screening_score"] = p[2]
  result_file = os.path.join(config["save_path"], config["data_index"])
  pathlib.Path(os.path.dirname(result_file)).mkdir(parents=True, exist_ok=True)
  inference_results.to_csv(result_file, float_format='%.8f', index=False)

  logger.info("Inference complete. Congratulations!")
