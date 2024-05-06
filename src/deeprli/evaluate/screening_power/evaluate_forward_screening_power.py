import os, argparse, json, pathlib
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def round_decimal_half_up(x, n):
  if n == 0:
    return Decimal(x).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
  elif n == 1:
    return Decimal(x).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
  elif n == 2:
    return Decimal(x).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
  elif n == 3:
    return Decimal(x).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
  else:
    raise Exception("Rounding is not supported.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate forward screening power")
  parser.add_argument("--target-info-index", type=str, help="the path of power_screening/TargetInfo.dat")
  parser.add_argument("--predicted-data-index", type=str, help="the path of an index file contains predicted values")
  parser.add_argument("--prediction-target", type=str, help="prediction target (pKd or DeltaG)")
  parser.add_argument("--is-ensemble", action="store_true", help="if the predicted-data-index includes predicted results of an ensemble or not")
  parser.add_argument("--save-path", type=str, help="the path to deposit summarized results")
  args = parser.parse_args()

  pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

  target_info_index = pd.read_csv(
    args.target_info_index,
    sep="[,,\t, ]+",
    comment='#',
    header=None,
    names=["T", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"],
    engine="python"
  )

  target_info_index = target_info_index.set_index('T')
  target_info_index = target_info_index.sort_index()

  target_list = list(target_info_index.index)

  decoy_list = []
  for i in np.arange(1, 11):
    decoy_list.extend(target_info_index['L' + str(i)].tolist())
  decoy_list = list(filter(None, list(set(decoy_list))))

  t0 = int(round_decimal_half_up(len(decoy_list) * 0.005, 0))
  t1 = int(round_decimal_half_up(len(decoy_list) * 0.01, 0))
  t5 = int(round_decimal_half_up(len(decoy_list) * 0.05, 0))
  t10 = int(round_decimal_half_up(len(decoy_list) * 0.10, 0))

  if t0 < 1:
    print("The number of top 0.5%% ligands is less than 1")
    print("In this case, we set the cutoff of top 0.5%% = 1")
    t0 = 1
  if t1 < 1:
    print("The number of top 1%% ligands is less than 1")
    print("In this case, we set the cutoff of top 1%% = 1")
    t1 = 1
  if t5 < 1:
    print("The number of top 5%% ligands is less than 1")
    print("In this case, we set the cutoff of top 5%% = 1")
    t5 = 1
  if t10 < 1:
    print("The number of top 10%% ligands is less than 1")
    print("In this case, we set the cutoff of top 10%% = 1")
    t10 = 1

  Top0 = pd.DataFrame(index=target_list, columns=["success"])
  Top1 = pd.DataFrame(index=target_list, columns=["success"])
  Top5 = pd.DataFrame(index=target_list, columns=["success"])
  Top10 = pd.DataFrame(index=target_list, columns=["success"])
  EF0 = pd.DataFrame(index=target_list, columns=["enrichment"])
  EF1 = pd.DataFrame(index=target_list, columns=["enrichment"])
  EF5 = pd.DataFrame(index=target_list, columns=["enrichment"])
  EF10 = pd.DataFrame(index=target_list, columns=["enrichment"])

  forward_screening_results = pd.DataFrame(index=range(len(target_list)), columns=["target"] + list(range(t10)))

  predicted_data_index = pd.read_csv(args.predicted_data_index)
  complex_path_split = predicted_data_index["complex_path"].str.split('/', expand=True)
  predicted_data_index["receptor_id"] = complex_path_split.iloc[:, -3]
  predicted_data_index["ligand_id"] = complex_path_split.iloc[:, -2].str.split('-', expand=True)[1]
  predicted_data_index["pose_idx"] = complex_path_split.iloc[:, -1].str.split('_', expand=True)[1]
  if args.is_ensemble:
    if args.prediction_target == "pKd":
      predicted_data_index.rename(columns={"pKd_pred_mean": "pKd_pred"}, inplace=True)
    elif args.prediction_target == "DeltaG":
      predicted_data_index.rename(columns={"DeltaG_pred_mean": "DeltaG_pred"}, inplace=True)

  for i, target_code in enumerate(target_info_index.index):
    target_results = predicted_data_index[predicted_data_index["receptor_id"] == target_code]
    if args.prediction_target == "pKd":
      target_results_grouped = pd.DataFrame(target_results.groupby("ligand_id")["pKd_pred"].max())
      target_results_grouped_sorted = target_results_grouped.sort_values("pKd_pred", ascending=False)
    elif args.prediction_target == "DeltaG":
      target_results_grouped = pd.DataFrame(target_results.groupby("ligand_id")["DeltaG_pred"].min())
      target_results_grouped_sorted = target_results_grouped.sort_values("DeltaG_pred")

    for j in range(t10):
      forward_screening_results.loc[i][j] = target_results_grouped_sorted.index[j]
    forward_screening_results.loc[i]["target"] = target_code
    top_ligand = target_info_index.loc[target_code]["L1"]
    all_active_lig = list(target_info_index.loc[target_code].dropna())
    ntb_total = len(all_active_lig)
    for l, k in zip(['0', '1', '5', '10'], [t0, t1, t5, t10]):
      candidates = list(target_results_grouped_sorted.index[0:k])
      Top = locals()[f"Top{l}"]
      if top_ligand in candidates:
        Top.loc[target_code]["success"] = 1
      else:
        Top.loc[target_code]["success"] = 0

      EF = locals()[f"EF{l}"]
      ntb = 0
      for lig in all_active_lig:
        if lig in candidates:
          ntb += 1
        else:
          continue
      if l == '0':
        efvalue = ntb / (ntb_total * 0.5 * 0.01)
      else:
        efvalue = ntb / (ntb_total * int(l) * 0.01)
      EF.loc[target_code]["enrichment"] = efvalue

  # calculate success rates and enrichment factors
  top0success = Top0['success'].sum() / len(Top0) * 100
  top1success = Top1['success'].sum() / len(Top1) * 100
  top5success = Top5['success'].sum() / len(Top5) * 100
  top10success = Top10['success'].sum() / len(Top10) * 100
  ef0factor = EF0['enrichment'].sum() / len(EF0)
  ef1factor = EF1['enrichment'].sum() / len(EF1)
  ef5factor = EF5['enrichment'].sum() / len(EF5)
  ef10factor = EF10['enrichment'].sum() / len(EF10)

  performance_indicators = {
    "ef0.5factor": ef0factor,
    "ef1factor": ef1factor,
    "ef5factor": ef5factor,
    "ef10factor": ef10factor,
    "top0.5success": top0success,
    "top1success": top1success,
    "top5success": top5success,
    "top10success": top10success
  }

  Top0.index.name = "target"
  Top1.index.name = "target"
  Top5.index.name = "target"
  Top10.index.name = "target"
  EF0.index.name = "target"
  EF1.index.name = "target"
  EF5.index.name = "target"
  EF10.index.name = "target"

  Top0.to_csv(os.path.join(args.save_path, "top0.5_success.csv"))
  Top1.to_csv(os.path.join(args.save_path, "top1_success.csv"))
  Top5.to_csv(os.path.join(args.save_path, "top5_success.csv"))
  Top10.to_csv(os.path.join(args.save_path, "top10_success.csv"))
  EF0.to_csv(os.path.join(args.save_path, "ef0.5_factor.csv"), float_format='%.8f')
  EF1.to_csv(os.path.join(args.save_path, "ef1_factor.csv"), float_format='%.8f')
  EF5.to_csv(os.path.join(args.save_path, "ef5_factor.csv"), float_format='%.8f')
  EF10.to_csv(os.path.join(args.save_path, "ef10_factor.csv"), float_format='%.8f')


  print("Performance:", performance_indicators)
  with open(os.path.join(args.save_path, "performance_indicators.json"), "wt") as f:
    json.dump(performance_indicators, f)