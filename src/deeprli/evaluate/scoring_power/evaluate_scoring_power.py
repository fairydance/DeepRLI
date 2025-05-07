import os, argparse, json, pathlib
import pandas as pd

from deeprli.utils.metrics import mse, rmse, pearson, spearman, ci


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate scoring power")
  parser.add_argument("--true-data-index", type=str, help="the path of an index file contains true values")
  parser.add_argument("--predicted-data-index", type=str, help="the path of an index file contains predicted values")
  parser.add_argument("--prediction-target", type=str, default="pKd", help="prediction target (pKd or DeltaG)")
  parser.add_argument("--is-ensemble", action="store_true", help="if the predicted-data-index includes predicted results of an ensemble or not")
  parser.add_argument("--save-path", type=str, help="the path to deposit results")
  args = parser.parse_args()

  pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

  true_data_index = pd.read_csv(args.true_data_index)
  y = true_data_index[args.prediction_target].to_numpy()

  predicted_data_index = pd.read_csv(args.predicted_data_index)
  if not args.is_ensemble:
    p = predicted_data_index[f"{args.prediction_target}_pred"].to_numpy()
  else:
    p = predicted_data_index[f"{args.prediction_target}_pred_mean"].to_numpy()
  
  msev = mse(p, y)
  rmsev = rmse(p, y)
  pearsonv = pearson(p, y)
  spearmanv = spearman(p, y)
  civ = ci(p, y)

  performance_indicators = {"mse": msev, "rmse": rmsev, "pearson": pearsonv, "spearman": spearmanv, "ci": civ}
  print("Performance:", performance_indicators)
  with open(os.path.join(args.save_path, "performance_indicators.json"), "wt") as f:
    json.dump(performance_indicators, f)
