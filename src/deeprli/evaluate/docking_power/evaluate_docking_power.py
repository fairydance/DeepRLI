import os, argparse, json, pathlib
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate docking power")
  parser.add_argument("--core-set-index", type=str, help="the path of power_docking/CoreSet.dat")
  parser.add_argument("--predicted-data-index", type=str, help="the path of an index file contains predicted values")
  parser.add_argument("--prediction-target", type=str, help="prediction target (pKd or DeltaG)")
  parser.add_argument("--is-ensemble", action="store_true", help="if the predicted-data-index includes predicted results of an ensemble or not")
  parser.add_argument("--save-path", type=str, help="the path to deposit summarized results")
  args = parser.parse_args()

  pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

  core_set_index = pd.read_csv(
    args.core_set_index,
    sep="[,,\t, ]+",
    comment='#',
    header=None,
    names=["code", "resl", "year", "logka", "Ka", "target"],
    engine="python"
  )

  target_codes = sorted(list(core_set_index["code"]))

  Top1 = pd.DataFrame(index=target_codes, columns=["success"])
  Top2 = pd.DataFrame(index=target_codes, columns=["success"])
  Top3 = pd.DataFrame(index=target_codes, columns=["success"])
  SP2 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP3 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP4 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP5 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP6 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP7 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP8 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP9 = pd.DataFrame(index=target_codes, columns=["spearman"])
  SP10 = pd.DataFrame(index=target_codes, columns=["spearman"])
  docking_results = pd.DataFrame(index=range(len(target_codes)), columns=["code", "pose_idx_0", "rmsd_0", "pose_idx_1", "rmsd_1", "pose_idx_2", "rmsd_2"])

  predicted_data_index = pd.read_csv(args.predicted_data_index)
  complex_path_split = predicted_data_index["complex_path"].str.split('/', expand=True)
  predicted_data_index["complex_id"] = complex_path_split.iloc[:, -2]
  predicted_data_index["pose_idx"] = complex_path_split.iloc[:, -1].str.split('_', expand=True)[1]
  if args.is_ensemble:
    if args.prediction_target == "pKd":
      predicted_data_index.rename(columns={"pKd_pred_mean": "pKd_pred"}, inplace=True)
    elif args.prediction_target == "DeltaG":
      predicted_data_index.rename(columns={"DeltaG_pred_mean": "DeltaG_pred"}, inplace=True)

  for i, target_code in enumerate(target_codes):
    target_results = predicted_data_index[predicted_data_index["complex_id"] == target_code]
    if args.prediction_target == "pKd":
      target_results_sorted = target_results.sort_values("pKd_pred", ascending=False)
    elif args.prediction_target == "DeltaG":
      target_results_sorted = target_results.sort_values("DeltaG_pred")
    for j in range(3):
      docking_results.loc[i][f"pose_idx_{j}"] = target_results_sorted.iloc[j]["pose_idx"]
      docking_results.loc[i][f"rmsd_{j}"] = target_results_sorted.iloc[j]["rmsd"]
    docking_results.loc[i]["code"] = target_code
    for j in range(1, 4):
      rmsd_min = target_results_sorted.iloc[:j]["rmsd"].min()
      Top = locals()[f"Top{j}"]
      if rmsd_min <= 2:
        Top.loc[target_code]["success"] = 1
      else:
        Top.loc[target_code]["success"] = 0
    for j in range(2, 11):
      target_results_filtered = target_results[target_results["rmsd"] <= j]
      SP = locals()[f"SP{j}"]
      if len(target_results_filtered) >= 5:
        if args.prediction_target == "pKd":
          SP.loc[target_code]["spearman"] = np.negative(target_results_filtered.corr("spearman", numeric_only=True)["rmsd"]["pKd_pred"])
        elif args.prediction_target == "DeltaG":
          SP.loc[target_code]["spearman"] = target_results_filtered.corr("spearman", numeric_only=True)["rmsd"]["DeltaG_pred"]
      else:
        continue

  SP2 = SP2.dropna(subset=["spearman"])
  SP3 = SP3.dropna(subset=["spearman"])
  SP4 = SP4.dropna(subset=["spearman"])
  SP5 = SP5.dropna(subset=["spearman"])
  SP6 = SP6.dropna(subset=["spearman"])
  SP7 = SP7.dropna(subset=["spearman"])
  SP8 = SP8.dropna(subset=["spearman"])
  SP9 = SP9.dropna(subset=["spearman"])
  SP10 = SP10.dropna(subset=["spearman"])

  top1success = Top1["success"].sum() / len(Top1) * 100
  top2success = Top2["success"].sum() / len(Top2) * 100
  top3success = Top3["success"].sum() / len(Top3) * 100
  sp2 = SP2["spearman"].sum() / len(SP2)
  sp3 = SP3["spearman"].sum() / len(SP3)
  sp4 = SP4["spearman"].sum() / len(SP4)
  sp5 = SP5["spearman"].sum() / len(SP5)
  sp6 = SP6["spearman"].sum() / len(SP6)
  sp7 = SP7["spearman"].sum() / len(SP7)
  sp8 = SP8["spearman"].sum() / len(SP8)
  sp9 = SP9["spearman"].sum() / len(SP9)
  sp10 = SP10["spearman"].sum() / len(SP10)

  performance_indicators = {
    "top1success": top1success,
    "top2success": top2success,
    "top3success": top3success,
    "sp2": sp2,
    "sp3": sp3,
    "sp4": sp4,
    "sp5": sp5,
    "sp6": sp6,
    "sp7": sp7,
    "sp8": sp8,
    "sp9": sp9,
    "sp10": sp10,
  }

  print("Performance:", performance_indicators)
  with open(os.path.join(args.save_path, "performance_indicators.json"), "wt") as f:
    json.dump(performance_indicators, f)

  Top1.index.name = "target"
  Top2.index.name = "target"
  Top3.index.name = "target"
  Top1.to_csv(os.path.join(args.save_path, "top1_success.csv"))
  Top2.to_csv(os.path.join(args.save_path, "top2_success.csv"))
  Top3.to_csv(os.path.join(args.save_path, "top3_success.csv"))
  docking_results.to_csv(os.path.join(args.save_path, "docking_results.csv"), float_format='%.2f', index=False)