import os, sys, time, argparse, pathlib, logging, json, shutil, pickle
from distutils.util import strtobool
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
import torch

from deeprli.datasets import ReceptorDataset, LigandDataset, LightDatasetForInfer
from deeprli.model import DeepRLI, DeepRLIForInterpretation

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Use a protein in PDB format and multiple ligands in SDF format as input to predict scores.")
  parser.add_argument('--config', type=str, help="the path of a configuration file")
  parser.add_argument("--instance-name", type=str, help="the name of this instance")
  parser.add_argument("--receptor-file-path", type=str, help="the path of a protein file in PDB format")
  parser.add_argument("--ligand-file-path", type=str, help="the path of a ligand file in SDF format")
  parser.add_argument("--known-ligand-file-path", type=str, help="the path of a known ligand file in SDF format")
  parser.add_argument("--dist-cutoff", type=float, help="distance cutoff for edge assignment")
  parser.add_argument("--bs-cutoff", type=float, help="distance cutoff for residues around the known ligands")
  parser.add_argument("--model", type=str, help="the path of a trained model or an ensemble of trained models")
  parser.add_argument("--model-format", type=str, help="state_dict or model_object")
  parser.add_argument("--interpretation", type=lambda x: bool(strtobool(x)), help="with interpretation or not")
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
  args = parser.parse_args()

  if args.config and os.path.isfile(args.config):
    with open(args.config) as f:
      config = json.load(f)
  else:
    config = {}

  config_keys = [
    "instance_name",
    "receptor_file_path",
    "ligand_file_path",
    "known_ligand_file_path",
    "dist_cutoff",
    "bs_cutoff",
    "model",
    "model_format",
    "interpretation",
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
  ]

  config.update({key: value for key in config_keys if (value:=getattr(args, key, None)) is not None})
  if config.get("instance_name") is None: config["instance_name"] = "default"
  if config.get("receptor_file_path") is None: config["receptor_file_path"] = "receptor.pdb"
  if config.get("ligand_file_path") is None: config["ligand_file_path"] = "ligands.sdf"
  if config.get("known_ligand_file_path") is None: config["known_ligand_file_path"] = "known_ligands.sdf"
  if config.get("dist_cutoff") is None: config["dist_cutoff"] = 6.5
  if config.get("bs_cutoff") is None: config["bs_cutoff"] = 7.5
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

  pathlib.Path(config["save_path"]).mkdir(parents=True, exist_ok=True)

  logging.root.handlers = []
  logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
      logging.FileHandler(os.path.join(config["save_path"], "prediction.log"), 'w'),
      logging.StreamHandler(sys.stdout)
    ]
  )

  logger = logging.getLogger(__name__)
  logger.info("DeepRLI Prediction Records")

  now = time.localtime()
  timestamp = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
  logger.info(f"Date: {timestamp}")
  logger.info(f"Config: {config}")

  # build the preset data directory
  pathlib.Path(os.path.join(config["save_path"], "data", "index", config["instance_name"])).mkdir(parents=True, exist_ok=True)
  pathlib.Path(os.path.join(config["save_path"], "data", "raw", config["instance_name"], "ligands")).mkdir(parents=True, exist_ok=True)
  
  shutil.copy2(config["receptor_file_path"], os.path.join(config["save_path"], "data", "raw", config["instance_name"], "receptor.pdb"))
  
  known_ligands = Chem.SDMolSupplier(config["known_ligand_file_path"])
  known_ligand_names = []
  for ligand in known_ligands:
    name = str(ligand.GetProp("_Name"))
    known_ligand_names.append(f"{name}.known")
    with Chem.SDWriter(os.path.join(config["save_path"], "data", "raw", config["instance_name"], "ligands", f"{name}.known.sdf")) as w:
      w.write(ligand)
  receptor_index_df = pd.DataFrame([[config["instance_name"], known_ligand_names]], columns=["complex_path", "known_ligands"])
  receptor_index_df.to_csv(os.path.join(config["save_path"], "data", "index", config["instance_name"], "receptor.csv"), index=False)

  ligands = Chem.SDMolSupplier(config["ligand_file_path"])
  ligand_names = []
  for ligand in ligands:
    name = str(ligand.GetProp("_Name"))
    ligand_names.append(f'{config["instance_name"]}/{name}')
    with Chem.SDWriter(os.path.join(config["save_path"], "data", "raw", config["instance_name"], "ligands", f"{name}.sdf")) as w:
      w.write(ligand)
  ligand_index_df = pd.DataFrame({"complex_path": ligand_names})
  ligand_index_df.to_csv(os.path.join(config["save_path"], "data", "index", config["instance_name"], "ligands.csv"), index=False)

  # preprocess
  ReceptorDataset(
    root=os.path.join(config["save_path"], "data"),
    data_index=f'index/{config["instance_name"]}/receptor.csv',
    ligand_file_types=["sdf"],
    dist_cutoff=config["dist_cutoff"],
    bs_cutoff=config["bs_cutoff"]
  )

  LigandDataset(
    root=os.path.join(config["save_path"], "data"),
    data_index=f'index/{config["instance_name"]}/ligands.csv',
    ligand_file_types=["sdf"],
    dist_cutoff=config["dist_cutoff"]
  )

  # compile data
  pathlib.Path(os.path.join(config["save_path"], "data", "compiled", config["instance_name"])).mkdir(parents=True, exist_ok=True)
  shutil.copy2(os.path.join(config["save_path"], "data", "processed", config["instance_name"], "receptor.pkl"),
               os.path.join(config["save_path"], "data", "compiled", config["instance_name"], "receptor.processed.pkl"))

  ligand_data = {}
  ligand_processed_index_df = pd.read_csv(os.path.join(config["save_path"], "data", "index", config["instance_name"], "ligands.processed.csv"))
  for i, row in tqdm(ligand_processed_index_df.iterrows(), total=ligand_processed_index_df.shape[0]):
    with open(os.path.join(config["save_path"], "data", "processed", config["instance_name"], "ligands", f'{row["complex_path"].split("/")[-1]}.pkl'), "rb") as f:
      ligand_data[row["complex_path"]] = pickle.load(f)

  with open(os.path.join(config["save_path"], "data", "compiled", config["instance_name"], "ligands.processed.pkl"), "wb") as f:
    pickle.dump(ligand_data, f)

  # inference
  device = torch.device(f"cuda:{config['gpu_id']}" if config.get("gpu_id") is not None and torch.cuda.is_available() else "cpu")

  logger.info("")
  logger.info(f"Device: " + ("GPU" if config.get("gpu_id") is not None and torch.cuda.is_available() else "CPU"))
  logger.info("=" * 80)
  logger.info(f"Inferring task on {device}")

  ## load data
  data = LightDatasetForInfer(root=os.path.join(config["save_path"], "data"), receptor_data_file=f'compiled/{config["instance_name"]}/receptor.processed.pkl',
                              ligand_data_index=f'index/{config["instance_name"]}/ligands.processed.csv', ligand_data_files=[f'compiled/{config["instance_name"]}/ligands.processed.pkl'])

  logger.info("")
  logger.info(f"Dataset Info")
  logger.info("=" * 80)
  logger.info(f"Dataset to be inferred: {data}")
  logger.info(f"> Size: {len(data)}")

  data_loader = torch.utils.data.DataLoader(data, batch_size=config["batch"], collate_fn=LightDatasetForInfer.collate_fn, num_workers=config["num_workers"])

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
        use_residual=config["use_residual"]
      )
      model.to(device)
      model.load_state_dict(torch.load(config["model"], map_location=device))

    model.eval()
    p = [[], [], []]
    for batch in tqdm(data_loader):
      batch = {k: v.to(device) for k, v in batch.items()}
      result = model(batch)
      for i in range(3):
        p[i].extend(result[i].tolist())
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
        use_residual=config["use_residual"]
      )
      model.to(device)
      model.load_state_dict(torch.load(config["model"], map_location=device))

    model.eval()
    p = [[], [], []]
    for j, batch in enumerate(tqdm(data_loader)):
      batch = {k: v.to(device) for k, v in batch.items()}
      result = model(batch)
      torch.save(result, os.path.join(config["save_path"], "results", f"result_{j}.pth"))
      for i in range(3):
        p[i].extend(result["scores"][i].tolist())

  inference_results = pd.read_csv(os.path.join(config["save_path"], "data", "index", config["instance_name"], "ligands.processed.csv"))
  inference_results["scoring_score"] = p[0]
  inference_results["docking_score"] = p[1]
  inference_results["screening_score"] = p[2]
  result_file = os.path.join(config["save_path"], "results", "index", config["instance_name"], "ligands.processed.csv")
  pathlib.Path(os.path.dirname(result_file)).mkdir(parents=True, exist_ok=True)
  inference_results.to_csv(result_file, float_format='%.8f', index=False)

  logger.info("Prediction complete. Congratulations!")