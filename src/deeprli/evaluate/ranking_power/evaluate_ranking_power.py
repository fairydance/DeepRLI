import os, argparse, json, pathlib
import numpy as np
import pandas as pd

def calculate_predictive_index(df):
	dfsorted = df.sort_values(["pKd_true"], ascending=True)
	W = []
	WC = []
	lst = list(dfsorted.index)
	for i in np.arange(0, 5):
		xi = lst[i]
		score = float(dfsorted.loc[xi]["pKd_pred"])
		bindaff = float(dfsorted.loc[xi]["pKd_true"])
		for j in np.arange(i+1, 5):
			xj = lst[j]
			scoretemp = float(dfsorted.loc[xj]["pKd_pred"])
			bindafftemp = float(dfsorted.loc[xj]["pKd_true"])
			w_ij = abs(bindaff-bindafftemp)
			W.append(w_ij)
			if score < scoretemp:
				WC.append(w_ij)
			elif score > scoretemp:
				WC.append(-w_ij)
			else:
				WC.append(0)
	pi = float(sum(WC))/float(sum(W))
	return pi


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate ranking power")
  parser.add_argument("--core-set-index", type=str, help="the path of power_ranking/CoreSet.dat")
  parser.add_argument("--predicted-data-index", type=str, help="the path of an index file contains predicted values")
  parser.add_argument("--is-ensemble", action="store_true", help="if the predicted-data-index includes predicted results of an ensemble or not")
  parser.add_argument("--save-path", type=str, help="the path to deposit summarized results")
  args = parser.parse_args()

  pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

  core_set_index = pd.read_csv(
    args.core_set_index,
    sep="[,,\t, ]+",
    comment='#',
    header=None,
    names=["code", "resl", "year", "logKa", "Ka", "target"],
    engine="python"
  )

  core_set_index.drop(columns=["resl", "year", "Ka"], inplace=True)
  core_set_index.rename(columns={"code": "pdb_id", "logKa": "pKd_true"}, inplace=True)
  predicted_data_index = pd.read_csv(args.predicted_data_index)
  predicted_data_index["pdb_id"] = predicted_data_index["complex_path"].str.split('/').str[-1]
  # if not args.is_ensemble:
  #   predicted_data_index.rename(columns={"pKd": "pKd_pred"}, inplace=True)
  # else:
  #   predicted_data_index.rename(columns={"pKd_mean": "pKd_pred"}, inplace=True)
  if args.is_ensemble:
    predicted_data_index.rename(columns={"pKd_mean": "pKd_pred"}, inplace=True)
  predicted_data_index = predicted_data_index[["pdb_id", "pKd_pred"]]
  merged_data_index = pd.merge(core_set_index, predicted_data_index, on="pdb_id")
  merged_data_index_grouped = merged_data_index.groupby("target")

  target_list = merged_data_index_grouped.apply(lambda x: x.sort_values("pKd_true", ascending=False).iloc[0])["pdb_id"].tolist()

  spearman = pd.DataFrame(index=target_list, columns=["spearman"])
  kendall = pd.DataFrame(index=target_list, columns=["kendall"])
  pi = pd.DataFrame(index=target_list, columns=["pi"])
  ranking_results = pd.DataFrame(index=range(len(target_list)), columns=["target", "rank_0", "rank_1", "rank_2", "rank_3", "rank_4"])

  for i, (name, group) in enumerate(merged_data_index_grouped):
    group = group[["pdb_id", "pKd_true", "pKd_pred"]]
    group = group.sort_values("pKd_pred", ascending=False)
    target = group.sort_values("pKd_true", ascending=False).iloc[0]["pdb_id"]
    if len(group) == 5:
      spearman.loc[target]["spearman"] = group.corr("spearman", numeric_only=True)["pKd_true"]["pKd_pred"]
      kendall.loc[target]["kendall"] = group.corr("kendall", numeric_only=True)["pKd_true"]["pKd_pred"]
      pi.loc[target]["pi"] = calculate_predictive_index(df=group)
      for j in range(5):
        ranking_results.loc[i][f"rank_{j}"] = group.iloc[j]["pdb_id"]
      ranking_results.loc[i]["target"] = target
    else:
      spearman.drop(target, inplace=True)
      kendall.drop(target, inplace=True)
      pi.drop(target, inplace=True)

  spearman.to_csv(os.path.join(args.save_path, "spearman.csv"), index_label="target")
  kendall.to_csv(os.path.join(args.save_path, "kendall.csv"), index_label="target")
  pi.to_csv(os.path.join(args.save_path, "pi.csv"), index_label="target")

  spearman_mean = spearman["spearman"].sum() / len(spearman)
  kendall_mean = kendall["kendall"].sum() / len(kendall)
  pi_mean = pi["pi"].sum() / len(pi)

  performance_indicators = {"spearman": spearman_mean, "kendall": kendall_mean, "pi": pi_mean}
  print("Performance:", performance_indicators)
  with open(os.path.join(args.save_path, "performance_indicators.json"), "wt") as f:
    json.dump(performance_indicators, f)