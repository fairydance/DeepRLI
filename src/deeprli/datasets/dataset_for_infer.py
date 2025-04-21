import os, pickle, copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import networkx as nx
import dgl
import torch
from torch.utils.data import Dataset

from deeprli.utils import DictQueue, judge_noncovalent_interaction_type, one_hot_encoding

def e_init(c, n):
  def func(edges):
    offsets = torch.linspace(0, c, n, device=edges.data["distance"].device)
    width = offsets[1] - offsets[0]
    dist_rbf = torch.exp(-0.5 / torch.pow(width, 2) * torch.pow(edges.data["distance"][..., None] - offsets[None, ...], 2))
    return {"feature": torch.cat([edges.data["feature"], dist_rbf], -1), "envelope": 0.5 * (torch.cos(edges.data["distance"] * np.pi / c) + 1.0)}
  return func


class LightDatasetForInfer(Dataset):
  def __init__(self, root, data_index="index/ligands.csv", dist_cutoff=6.5, receptor_cache_maxlen=100):
    self.root = root
    self.data_index = data_index
    self.dist_cutoff = dist_cutoff
    self.receptors = DictQueue(receptor_cache_maxlen)

    self.data_index_df = pd.read_csv(os.path.join(root, data_index))

  @property
  def processed_dir(self):
    return os.path.join(self.root, "processed")
  
  def get_node_features(self, g):
    f = lambda d: [d["is_ligand"]] +\
      one_hot_encoding(d["symbol"], ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Met', 'Unk'], with_unknown=True) +\
      one_hot_encoding(d["hybridization"], ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"]) +\
      one_hot_encoding(d["formal_charge"], [-2, -1, 0, 1, 2, 3, 4]) +\
      one_hot_encoding(d["degree"], [0, 1, 2, 3, 4, 5]) +\
      [d["is_donor"], d["is_acceptor"], d["is_neg_ionizable"], d["is_pos_ionizable"], d["is_zn_binder"],
      d["is_aromatic"], d["is_hydrophobe"], d["is_lumped_hydrophobe"]] # (1, 11, 6, 7, 6, 8) --> total 39
    
    for n, d in g.nodes(data=True):
      d["feature"] = f(d)

  def get_edge_features(self, g):
    f = lambda d: [d["is_intermolecular"], d["is_covalent"]] + one_hot_encoding(d["bond_type"], ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])

    for m, n, d in g.edges(data=True):
      d["feature"] = f(d)

  def __len__(self):
    return len(self.data_index_df)

  def __getitem__(self, idx):
    index_row = self.data_index_df.iloc[idx]
    instance_name, ligand_name = index_row["instance_name"], index_row["ligand_name"]

    # Load receptor
    if instance_name in self.receptors:
      receptor = self.receptors[instance_name]
    else:
      with open(os.path.join(self.processed_dir, instance_name, "receptor.pkl"), "rb") as f:
        receptor = pickle.load(f)
        self.receptors.put(instance_name, receptor)

     # Load ligand
    with open(os.path.join(self.processed_dir,  instance_name, "ligands", f"{ligand_name}.pkl"), "rb") as f:
      ligand = pickle.load(f)

    # Initialize a graph
    g = nx.DiGraph() # complex
    g.add_nodes_from(ligand.graph.nodes(data=True))
    n1 = ligand.rdmol.GetNumAtoms()
    
    pocket_mapping = {}
    for idx, orig_idx in enumerate(ligand.pocket_atom_idxs):
      complex_idx = n1 + idx
      pocket_mapping[orig_idx] = complex_idx
      g.add_node(complex_idx, **receptor.graph.nodes[orig_idx])

    g.add_edges_from(ligand.graph.edges(data=True))

    for u, v, data in receptor.graph.edges(data=True):
      if u in pocket_mapping and v in pocket_mapping:
        shifted_u = pocket_mapping[u]
        shifted_v = pocket_mapping[v]
        g.add_edge(shifted_u, shifted_v, **data)

    for lig_idx in range(n1):
      for pock_idx, orig_pock_idx in enumerate(ligand.pocket_atom_idxs):
        distance = ligand._pocket_distance_matrix[lig_idx][pock_idx]
        if distance < self.dist_cutoff:
          pock_complex_idx = pocket_mapping[orig_pock_idx]
          # Determine interaction type using node attributes from complex graph
          interaction_type = judge_noncovalent_interaction_type(g, lig_idx, pock_complex_idx)
          g.add_edge(lig_idx, pock_complex_idx, is_intermolecular=True, is_covalent=False,
            bond_type="NON-COVALENT", distance=distance, interaction_type=interaction_type)

    complex = SimpleNamespace()
    self.get_node_features(g)
    self.get_edge_features(g)
    complex.graph = dgl.from_networkx(g, node_attrs=["feature", "vdw_radii"], edge_attrs=["feature", "distance", "interaction_type"])

    data = {
      "complex.graph": complex.graph,
      "ligand.num_atoms": n1,
      "ligand.num_rotatable_bonds": ligand.num_rotatable_bonds,
      "receptor.num_atoms": len(ligand.pocket_atom_idxs),
    }

    data["complex.graph"] = dgl.add_reverse_edges(data["complex.graph"], copy_edata=True)

    return data
  
  @staticmethod
  def collate_fn(items):
    def make_batch_data(data_list):
      keys = ["complex.graph", "ligand.num_rotatable_bonds"]
      batch_data = {key: [data[key] for data in data_list] for key in keys}
      batch_data["complex.graph"] = dgl.batch(batch_data["complex.graph"])
      batch_data["complex.graph"].ndata["feature"] = batch_data["complex.graph"].ndata["feature"].float()
      batch_data["complex.graph"].edata["feature"] = batch_data["complex.graph"].edata["feature"].float()
      batch_data["complex.graph"].apply_edges(e_init(6.5, 33))
      batch_data["ligand.num_rotatable_bonds"] = torch.Tensor(batch_data["ligand.num_rotatable_bonds"])
      return batch_data

    return make_batch_data(items)
  

class HeavyDatasetForInfer(Dataset):
  def __init__(self, root=None, data_index="index/data.csv", data_file="compiled/data.pkl"):
    self.root = root
    self.data_index_df = pd.read_csv(os.path.join(root, data_index))

    with open(os.path.join(root, data_file), "rb") as f:
      self.data = pickle.load(f)

  def __len__(self):
    return len(self.data_index_df)

  def __getitem__(self, idx):
    data = copy.deepcopy(self.data[self.data_index_df["complex_path"][idx]])
    data["complex.graph"] = dgl.add_reverse_edges(data["complex.graph"], copy_edata=True)
    return data
  
  @staticmethod
  def collate_fn(items):
    def make_batch_data(data_list):
      keys = ["complex.graph", "ligand.num_rotatable_bonds"]
      batch_data = {key: [data[key] for data in data_list] for key in keys}
      batch_data["complex.graph"] = dgl.batch(batch_data["complex.graph"])
      batch_data["complex.graph"].ndata["feature"] = batch_data["complex.graph"].ndata["feature"].float()
      batch_data["complex.graph"].edata["feature"] = batch_data["complex.graph"].edata["feature"].float()
      batch_data["complex.graph"].apply_edges(e_init(6.5, 33))
      batch_data["ligand.num_rotatable_bonds"] = torch.Tensor(batch_data["ligand.num_rotatable_bonds"])
      return batch_data

    return make_batch_data(items)
