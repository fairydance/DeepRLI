import os, pickle, copy
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import networkx as nx
import dgl
import torch
from torch.utils.data import Dataset

from deeprli.base import Attributes, ChemicalElements
from deeprli.utils import one_hot_encoding, judge_noncovalent_interaction_type

def e_init(c, n):
  def func(edges):
    offsets = torch.linspace(0, c, n, device=edges.data["distance"].device)
    width = offsets[1] - offsets[0]
    dist_rbf = torch.exp(-0.5 / torch.pow(width, 2) * torch.pow(edges.data["distance"][..., None] - offsets[None, ...], 2))
    return {"feature": torch.cat([edges.data["feature"], dist_rbf], -1), "envelope": 0.5 * (torch.cos(edges.data["distance"] * np.pi / c) + 1.0)}
  return func


class LightDatasetForInfer(Dataset):
  def __init__(self, root, receptor_data_file="compiled/receptor.pkl", ligand_data_index="index/ligands.csv", ligand_data_files=["compiled/actives.pkl", "compiled/decoys.pkl"], dist_cutoff=6.5):
    self.root = root

    with open(os.path.join(root, receptor_data_file), "rb") as f:
      self.receptor = pickle.load(f)

    self.data_index_df = pd.read_csv(os.path.join(root, ligand_data_index))

    self.ligands = {}
    for ligand_data_file in ligand_data_files:
      with open(os.path.join(self.root, ligand_data_file), "rb") as f:
        self.ligands.update(pickle.load(f))

    self.dist_cutoff = dist_cutoff

    self.vdw_radii = {'C': 2.0, 'N': 1.7, 'O': 1.6, 'F': 1.5, 'Si': 2.2, 'P': 2.1, 'S': 2.0, 'Cl': 1.8, 'Br': 2.0,
                      'I': 2.2, 'At': 2.3, 'Met': 1.2, 'Unk': 1.2}

  @property
  def processed_dir(self):
    return os.path.join(self.root, "processed")

  @property
  def compiled_dir(self):
    return os.path.join(self.root, "compiled")
  
  def modify_symbol(self, symbol, mode=0):
    '''map atom symbol to allowable type'''
    if symbol in ChemicalElements.metals:
      return 'Met'
    if mode != 0:
      if symbol == 'Se':
        return 'S'
      elif symbol not in ('C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I', 'At'):
        return 'Unk'
    return symbol
  
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
    receptor = copy.deepcopy(self.receptor)
    ligand = copy.deepcopy(self.ligands[self.data_index_df["complex_path"][idx]])
    try:
      pocket_pdb_lines = receptor.pocket_pdb_block.split('\n')
    except:
      pocket_pdb_lines = receptor.pdb_block.split('\n')

    pocket_atom_idxs = []
    ATOM, resSeq_pre, resSeq, res_atom_idxs, coordinates = False, None, None, [], None
    for i, line in enumerate(pocket_pdb_lines):
      if (line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] != "HOH")) and line[76:78].strip() != 'H':
        ATOM = True
        resSeq = int(line[22:26])
        if resSeq != resSeq_pre:
          if resSeq_pre:
            min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
            if min_dist < self.dist_cutoff:
              pocket_atom_idxs.extend(res_atom_idxs)
          res_atom_idxs.append(i)
          coordinates = np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)
        else:
          res_atom_idxs.append(i)
          coordinates = np.concatenate((coordinates, np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)))
        resSeq_pre = resSeq
    if ATOM:
      min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
      if min_dist < self.dist_cutoff:
        pocket_atom_idxs.extend(res_atom_idxs)
    pocket_atom_idxs = set(pocket_atom_idxs)

    ## Initialize A Graph
    g = nx.DiGraph() # complex

    ### Create Ligand Nodes
    n1 = ligand.rdmol.GetNumAtoms()
    for i in range(n1):
      g.add_node(i, is_ligand=True, symbol=self.modify_symbol(ligand.symbols[i]),
        vdw_radii=self.vdw_radii[self.modify_symbol(ligand.symbols[i], mode=1)],
        hybridization=ligand.rdmol.GetAtomWithIdx(i).GetHybridization().name,
        formal_charge=ligand.rdmol.GetAtomWithIdx(i).GetFormalCharge(),
        degree=ligand.rdmol.GetAtomWithIdx(i).GetDegree(),
        is_donor=True if i in ligand.feature_dict["Donor"] else False,
        is_acceptor=True if i in ligand.feature_dict["Acceptor"] else False,
        is_neg_ionizable=True if i in ligand.feature_dict["NegIonizable"] else False,
        is_pos_ionizable=True if i in ligand.feature_dict["PosIonizable"] else False,
        is_zn_binder=True if i in ligand.feature_dict["ZnBinder"] else False,
        is_aromatic=True if i in ligand.feature_dict["Aromatic"] else False,
        is_hydrophobe=True if i in ligand.feature_dict["Hydrophobe"] else False,
        is_lumped_hydrophobe=True if i in ligand.feature_dict["LumpedHydrophobe"] else False)
      
    ### Create Receptor Nodes
    n2 = receptor.rdmol.GetNumAtoms()
    for i in range(n2):
      if i in pocket_atom_idxs:
        g.add_node(i + n1, is_ligand=False, symbol=self.modify_symbol(receptor.symbols[i]),
          vdw_radii=self.vdw_radii[self.modify_symbol(receptor.symbols[i], mode=1)],
          hybridization=receptor.rdmol.GetAtomWithIdx(i).GetHybridization().name,
          formal_charge=receptor.rdmol.GetAtomWithIdx(i).GetFormalCharge(),
          degree=receptor.rdmol.GetAtomWithIdx(i).GetDegree(),
          is_donor=True if i in receptor.feature_dict["Donor"] else False,
          is_acceptor=True if i in receptor.feature_dict["Acceptor"] else False,
          is_neg_ionizable=True if i in receptor.feature_dict["NegIonizable"] else False,
          is_pos_ionizable=True if i in receptor.feature_dict["PosIonizable"] else False,
          is_zn_binder=True if i in receptor.feature_dict["ZnBinder"] else False,
          is_aromatic=True if i in receptor.feature_dict["Aromatic"] else False,
          is_hydrophobe=True if i in receptor.feature_dict["Hydrophobe"] else False,
          is_lumped_hydrophobe=True if i in receptor.feature_dict["LumpedHydrophobe"] else False)
      
    ligand_ligand_distance_matrix = distance_matrix(ligand.positions, ligand.positions).tolist()
    receptor_receptor_distance_matrix = receptor.distance_matrix
    ligand_receptor_distance_matrix = distance_matrix(ligand.positions, receptor.positions).tolist()

    ### Create Ligand Edges
    for i in range(n1):
      for j in range(i + 1, n1):
        distance = ligand_ligand_distance_matrix[i][j]
        bond = ligand.rdmol.GetBondBetweenAtoms(i, j)
        if bond is not None:
          g.add_edge(i, j, is_intermolecular=False, is_covalent=True, bond_type=bond.GetBondType().name, distance=distance, interaction_type=[False, False, False])
        elif distance < self.dist_cutoff:
          g.add_edge(i, j, is_intermolecular=False, is_covalent=False, bond_type="NON-COVALENT", distance=distance,
            interaction_type=judge_noncovalent_interaction_type(g, i, j))
          
    ### Create Receptor Edges
    for i in range(n2):
      for j in range(i + 1, n2):
        if i in pocket_atom_idxs and j in pocket_atom_idxs:
          distance = receptor_receptor_distance_matrix[i][j]
          bond = receptor.rdmol.GetBondBetweenAtoms(i, j)
          if bond is not None:
            g.add_edge(i + n1, j + n1, is_intermolecular=False, is_covalent=True, bond_type=bond.GetBondType().name, distance=distance, interaction_type=[False, False, False])
          elif distance < self.dist_cutoff:
            g.add_edge(i + n1, j + n1, is_intermolecular=False, is_covalent=False, bond_type="NON-COVALENT", distance=distance,
              interaction_type=judge_noncovalent_interaction_type(g, i + n1, j + n1))
            
    ### Create Ligand-Receptor Edges
    for i in range(n1):
      for j in range(n2):
        if j in pocket_atom_idxs:
          distance = ligand_receptor_distance_matrix[i][j]
          if distance < self.dist_cutoff:
            g.add_edge(i, j + n1, is_intermolecular=True, is_covalent=False, bond_type="NON-COVALENT", distance=distance,
              interaction_type=judge_noncovalent_interaction_type(g, i, j + n1))

    complex = Attributes()
    self.get_node_features(g)
    self.get_edge_features(g)
    complex.graph = dgl.from_networkx(g, node_attrs=["feature", "vdw_radii"], edge_attrs=["feature", "distance", "interaction_type"])

    data = {
      "complex.graph": complex.graph,
      "ligand.num_atoms": n1,
      "ligand.num_rotatable_bonds": ligand.num_rotatable_bonds,
      "receptor.num_atoms": n2,
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
