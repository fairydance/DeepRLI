import os, argparse, pathlib, pickle
import numpy as np
import pandas as pd

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate data split scheme")
  parser.add_argument("--data-root", type=str, default="dataset", help="root directory where the dataset should be saved")
  parser.add_argument("--data-index", type=str, default="index/data.csv", help="data index file paths relative to root")
  parser.add_argument("--num-fold", type=int, default=10, help="number of folds for cross validation")
  parser.add_argument("--save-path", type=str, default="results", help="the path to deposit data split schemes")
  args = parser.parse_args()

  print(f"number of folds: {args.num_fold}")
  data_index_file_list = args.data_index.split(',')
  if len(data_index_file_list) == 1:
    data_index_df = pd.read_csv(os.path.join(args.data_root, args.data_index))
    data_size = len(data_index_df.index)
    print(f"data size: {data_size}")
    data_index_split = np.array_split(np.random.permutation(data_size), args.num_fold)

    fold_index = np.arange(args.num_fold)
    for i in range(args.num_fold):
      excluded = args.num_fold - i - 1
      fold_index_residual = fold_index[fold_index != excluded]
      data_index_residual = np.array([], dtype=int)
      for j in fold_index_residual:
        data_index_residual = np.concatenate((data_index_residual, data_index_split[j]))
      data_split_scheme = {"train_data_indices": data_index_residual.tolist(), "validation_data_indices": data_index_split[excluded].tolist()}
      print(f"train size: {len(data_split_scheme['train_data_indices'])}, validation size: {len(data_split_scheme['validation_data_indices'])}")
      out_dir = os.path.join(args.save_path, f"split_{i}")
      pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
      data_split_scheme_file = os.path.join(out_dir, "data_split_scheme.pkl")
      with open(data_split_scheme_file, "wb") as f:
        pickle.dump(data_split_scheme, f)
  elif len(data_index_file_list) > 1:
    data_index_df_list, data_size_list = [], []
    for i, data_index_file in enumerate(data_index_file_list):
      data_index_df_list.append(pd.read_csv(os.path.join(args.data_root, data_index_file)))
      data_size_list.append(len(data_index_df_list[i].index))
    print(f"data size: {','.join([str(data_size) for data_size in data_size_list])}")
    data_size_accumulator, data_size_accumulated = 0, []
    for data_size in data_size_list:
      data_size_accumulator += data_size
      data_size_accumulated.append(data_size_accumulator)
    print(f"data size accumulated: {','.join([str(data_size) for data_size in data_size_accumulated])}")
    data_index_split_list = []
    for i in range(len(data_size_accumulated)):
      if i == 0:
        data_index_split_list.append(np.array_split(np.random.permutation(data_size_accumulated[i]), args.num_fold))
      else:
        data_index_split_list.append(np.array_split(np.random.permutation(np.arange(data_size_accumulated[i - 1], data_size_accumulated[i])), args.num_fold))

    fold_index = np.arange(args.num_fold)
    for i in range(args.num_fold):
      excluded = args.num_fold - i - 1
      fold_index_residual = fold_index[fold_index != excluded]
      data_index_residual = np.array([], dtype=int)
      data_index_excluded = np.array([], dtype=int)
      for data_index_split in data_index_split_list:
        for j in fold_index_residual:
          data_index_residual = np.concatenate((data_index_residual, data_index_split[j]))
        data_index_excluded = np.concatenate((data_index_excluded, data_index_split[excluded]))
      data_index_residual = np.random.permutation(data_index_residual)
      data_index_excluded = np.random.permutation(data_index_excluded)
      data_split_scheme = {"train_data_indices": data_index_residual.tolist(), "validation_data_indices": data_index_excluded.tolist()}
      print(f"train size: {len(data_split_scheme['train_data_indices'])}, validation size: {len(data_split_scheme['validation_data_indices'])}")
      out_dir = os.path.join(args.save_path, f"split_{i}")
      pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
      data_split_scheme_file = os.path.join(out_dir, "data_split_scheme.pkl")
      with open(data_split_scheme_file, "wb") as f:
        pickle.dump(data_split_scheme, f)
  else:
    raise Exception()
