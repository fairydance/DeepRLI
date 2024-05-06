import os, argparse, pathlib, pickle
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
  """ Examples:
  python compile_data.py --data-root ${DATA_ROOT} --data-index "index/crystal/general-set_HiQ/all_data.processed.csv" --save-path ${DATA_ROOT}/compiled/crystal_data.processed.pkl
  python compile_data.py --data-root ${DATA_ROOT} --data-index "index/crossdock/refined-set/all_data.processed.csv" --save-path ${DATA_ROOT}/compiled/crossdock_data.processed.pkl
  python compile_data.py --data-root ${DATA_ROOT} --data-index "index/redock/refined-set/all_data.processed.csv" --save-path ${DATA_ROOT}/compiled/redock_data.processed.pkl

  python compile_data.py --data-root ${DATA_ROOT} --data-index "index/crystal/core-set/all_data.processed.csv" --save-path ${DATA_ROOT}/compiled/crystal_data.processed.pkl
  python compile_data.py --data-root ${DATA_ROOT} --data-index "index/decoy.docking/core-set/all_data.processed.csv" --save-path ${DATA_ROOT}/compiled/decoy.docking_data.processed.pkl
  python compile_data.py --data-root ${DATA_ROOT} --data-index "index/decoy.screening/core-set/all_data.processed.csv" --save-path ${DATA_ROOT}/compiled/decoy.screening_data.processed.pkl
  """
  parser = argparse.ArgumentParser(description="Compile Data")
  parser.add_argument("--data-root", type=str, default="dataset", help="Root directory where the dataset should be saved")
  parser.add_argument("--data-index", type=str, default="index/data.csv", help="Data index file path relative to root")
  parser.add_argument("--save-path", type=str, help="the path to deposit compiled data")
  args = parser.parse_args()

  data = {}
  data_index_df = pd.read_csv(os.path.join(args.data_root, args.data_index))
  for i, row in tqdm(data_index_df.iterrows(), total=data_index_df.shape[0]):
    with open(os.path.join(args.data_root, "processed", f"{row['complex_path']}.pkl"), "rb") as f:
      data[row["complex_path"]] = pickle.load(f)

  save_dir = os.path.dirname(args.save_path)
  pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
  with open(args.save_path, "wb") as f:
    pickle.dump(data, f)

  print("Compile complete. Congratulations!")
