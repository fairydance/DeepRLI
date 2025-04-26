import os, pickle, copy
import numpy as np
import pandas as pd
from ast import literal_eval
import dgl
import torch
from torch.utils.data import Dataset


def e_init(c, n):
  def func(edges):
    offsets = torch.linspace(0, c, n, device=edges.data["distance"].device)
    width = offsets[1] - offsets[0]
    dist_rbf = torch.exp(-0.5 / torch.pow(width, 2) * torch.pow(edges.data["distance"][..., None] - offsets[None, ...], 2))
    return {"feature": torch.cat([edges.data["feature"], dist_rbf], -1), "envelope": 0.5 * (torch.cos(edges.data["distance"] * np.pi / c) + 1.0)}
  return func


class AffinityDatasetForTrain(Dataset):
  def __init__(self, root=None, data_index="index/data.csv", data_files=["processed/data.pkl"]):
    self.root = root
    self.data_index_df = pd.read_csv(os.path.join(root, data_index))

    self.data = {}
    for data_file in data_files:
      with open(os.path.join(self.root, data_file), "rb") as f:
        self.data.update(pickle.load(f))

  @property
  def processed_dir(self):
    return os.path.join(self.root, "processed")
  
  def __getitem__(self, idx):
    index_row = self.data_index_df.iloc[idx]
    data = copy.deepcopy(self.data[index_row["complex_path"]])
    data["complex.graph"] = dgl.add_reverse_edges(data["complex.graph"], copy_edata=True)
    data["DeltaG"] = index_row["DeltaG"].tolist()
    return data
  
  @staticmethod
  def collate_fn(items):
    def make_batch_data(data_list):
      keys = ["complex.graph", "ligand.num_rotatable_bonds", "DeltaG"]
      batch_data = {key: [data[key] for data in data_list] for key in keys}
      batch_data["complex.graph"] = dgl.batch(batch_data["complex.graph"])
      batch_data["complex.graph"].ndata["feature"] = batch_data["complex.graph"].ndata["feature"].float()
      batch_data["complex.graph"].edata["feature"] = batch_data["complex.graph"].edata["feature"].float()
      batch_data["complex.graph"].apply_edges(e_init(6.5, 33))
      batch_data["ligand.num_rotatable_bonds"] = torch.Tensor(batch_data["ligand.num_rotatable_bonds"])
      batch_data["DeltaG"] = torch.Tensor(batch_data["DeltaG"])
      return batch_data
    
    return make_batch_data(items)
  

class ContrastiveDatasetForTrain(Dataset):
  def __init__(self, root=None, data_index="index/data.csv", data_files=["processed/data.pkl"]):
    self.root = root
    data_index = data_index.split(',')
    self.data_index_df_ctt = pd.read_csv(os.path.join(root, data_index[0]),
      converters={"positive_ids_redock": literal_eval, "negative_ids_redock": literal_eval, "negative_ids_crossdock": literal_eval})
    self.data_index_df_sup = pd.read_csv(os.path.join(root, data_index[1]))
    self.data_index_df_DeltaG = pd.read_csv(os.path.join(root, data_index[2]))
    self.data_index_df_DeltaG.set_index("complex_path", inplace=True)
    
    self.data = {}
    for data_file in data_files:
      with open(os.path.join(self.root, data_file), "rb") as f:
        self.data.update(pickle.load(f))

  @property
  def processed_dir(self):
    return os.path.join(self.root, "processed")

  def __len__(self):
    return len(self.data_index_df_ctt)

  def __getitem__(self, idx):
    target_id = self.data_index_df_ctt["anchor_id"][idx]
    anchor_id = np.random.choice(self.data_index_df_ctt["positive_ids_redock"][idx])
    negative_redock_id = np.random.choice(self.data_index_df_ctt["negative_ids_redock"][idx])
    negative_crossdock_id = np.random.choice(self.data_index_df_ctt["negative_ids_crossdock"][idx])
    crystal_data_tag = "crystal/refined-set"
    anchor_data_tag = "redock/refined-set"
    negative_redock_data_tag = "redock/refined-set"
    negative_crossdock_data_tag = "crossdock/refined-set"

    # def make_data(complex_path):
    #   with open(os.path.join(self.processed_dir, f"{complex_path}.pkl"), "rb") as f:
    #     data = pickle.load(f)
    #   data["complex.graph"] = dgl.add_reverse_edges(data["complex.graph"], copy_edata=True)
    #   if complex_path.split('/')[0] == "crystal":
    #     data["DeltaG"] = self.data_index_df_DeltaG.loc[complex_path, "DeltaG"].tolist()
    #   else:
    #     data["DeltaG"] = 0.0
    #   return data

    def make_data(complex_path):
      data = copy.deepcopy(self.data[complex_path])
      data["complex.graph"] = dgl.add_reverse_edges(data["complex.graph"], copy_edata=True)
      if complex_path.split('/')[0] == "crystal":
        data["DeltaG"] = self.data_index_df_DeltaG.loc[complex_path, "DeltaG"].tolist()
      else:
        data["DeltaG"] = 0.0
      return data

    crystal_data = make_data(f"{crystal_data_tag}/{target_id}")
    anchor_data = make_data(f"{anchor_data_tag}/{target_id}/{anchor_id}")
    negative_redock_data = make_data(f"{negative_redock_data_tag}/{target_id}/{negative_redock_id}")
    negative_crossdock_data = make_data(f"{negative_crossdock_data_tag}/{target_id}/{negative_crossdock_id}")
    random_data = make_data(np.random.choice(self.data_index_df_sup["complex_path"]))

    return (crystal_data, anchor_data, negative_redock_data, negative_crossdock_data, random_data)
  
  @staticmethod
  def collate_fn(items):
    def make_batch_data(data_list):
      keys = ["complex.graph", "ligand.num_rotatable_bonds", "DeltaG"]
      batch_data = {key: [data[key] for data in data_list] for key in keys}
      batch_data["complex.graph"] = dgl.batch(batch_data["complex.graph"])
      batch_data["complex.graph"].ndata["feature"] = batch_data["complex.graph"].ndata["feature"].float()
      batch_data["complex.graph"].edata["feature"] = batch_data["complex.graph"].edata["feature"].float()
      batch_data["complex.graph"].apply_edges(e_init(6.5, 33))
      batch_data["ligand.num_rotatable_bonds"] = torch.Tensor(batch_data["ligand.num_rotatable_bonds"])
      batch_data["DeltaG"] = torch.Tensor(batch_data["DeltaG"])
      return batch_data

    crystal_data_list, anchor_data_list, negative_redock_data_list, negative_crossdock_data_list, random_data_list = [], [], [], [], []
    for item in items:
      crystal_data_list.append(item[0])
      anchor_data_list.append(item[1])
      negative_redock_data_list.append(item[2])
      negative_crossdock_data_list.append(item[3])
      random_data_list.append(item[4])

    return (make_batch_data(crystal_data_list), make_batch_data(anchor_data_list), make_batch_data(negative_redock_data_list),
            make_batch_data(negative_crossdock_data_list), make_batch_data(random_data_list))