import os, pickle, logging, pathlib, json, gzip
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors
import networkx as nx
import dgl
import torch

from deeprli.data import Dataset
from deeprli.base import Attributes, ChemicalElements
from deeprli.utils import DictQueue, judge_noncovalent_interaction_type

logger = logging.getLogger(__name__)

class LigandDataset(Dataset):
  '''Ligand Dataset'''
  def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
               data_index="index/data.csv", ligand_file_types=["sdf"], dist_cutoff=6.5,
               receptor_cache_maxlen=100):
    self.data_index = data_index
    self.ligand_file_types = ligand_file_types
    self.dist_cutoff = dist_cutoff

    self.data_index_df = pd.read_csv(os.path.join(root, data_index))
    self.vdw_radii = {'C': 2.0, 'N': 1.7, 'O': 1.6, 'F': 1.5, 'Si': 2.2, 'P': 2.1, 'S': 2.0, 'Cl': 1.8, 'Br': 2.0,
                      'I': 2.2, 'At': 2.3, 'Met': 1.2, 'Unk': 1.2}
    
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    self.ptable = Chem.GetPeriodicTable()

    self.receptors = DictQueue(receptor_cache_maxlen)

    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
    return [str(Path(row["instance_name"]/"ligands"/row["ligand_name"])) for _, row in self.data_index_df.iterrows()]

  @property
  def processed_file_names(self):
    return [str(Path(row["instance_name"]/"ligands"/f'{row["ligand_name"]}.pkl')) for _, row in self.data_index_df.iterrows()]
  
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

  def extract_features(self, molecule):
    molecule.symbols = [atom.GetSymbol() for atom in molecule.rdmol.GetAtoms()]
    molecule.positions = molecule.rdmol.GetConformer().GetPositions()
    features = self.feature_factory.GetFeaturesForMol(molecule.rdmol)
    molecule.feature_dict = defaultdict(list)
    for feature in features:
      molecule.feature_dict[feature.GetFamily()].extend(feature.GetAtomIds())
  
  def data_maker(self, index_row):
    instance_name, ligand_name = index_row["instance_name"], index_row["ligand_name"]

    # Load receptor
    if instance_name in self.receptors:
      receptor = self.receptors[instance_name]
    else:
      with open(os.path.join(self.processed_dir, instance_name, "receptor.pkl"), "rb") as f:
        receptor = pickle.load(f)
        self.receptors.put(instance_name, receptor)
    
    # Parse ligand
    ligand = SimpleNamespace(rdmol=None)
    for ligand_file_type in self.ligand_file_types:
      ligand_file_path = os.path.join(self.raw_dir, instance_name, "ligands", f"{ligand_name}.{ligand_file_type}")
      if not os.path.exists(ligand_file_path):
        logger.info(f"[{instance_name}] Ligand File Not Found (.{ligand_file_type}): {ligand_file_path}")
      else:
        if ligand_file_type.split('.')[-1] == "sdf":
          ligand = Attributes({"rdmol": Chem.SDMolSupplier(ligand_file_path)[0]})
        elif ligand_file_type.split('.')[-1] == "mol2":
          ligand = Attributes({"rdmol": Chem.MolFromMol2File(ligand_file_path)})
        elif ligand_file_type.split('.')[-1] == "pdb":
          ligand = Attributes({"rdmol": Chem.MolFromPDBFile(ligand_file_path)})
        if ligand.rdmol is None:
          logger.info(f"[{instance_name}] Ligand Molecule Parsing Failed (.{ligand_file_type}): {ligand_file_path}")
        else:
          break

    if ligand.rdmol is None:
      return None

    ligand.rdmol = Chem.RemoveHs(ligand.rdmol)
    self.extract_features(ligand)
    ligand.num_rotatable_bonds = Descriptors.NumRotatableBonds(ligand.rdmol)
    ligand.distance_matrix = distance_matrix(ligand.positions, ligand.positions).tolist()
    ligand.graph = nx.DiGraph()

    # Create ligand nodes
    n = ligand.rdmol.GetNumAtoms()
    for i in range(n):
      ligand.graph.add_node(i, is_ligand=True, symbol=self.modify_symbol(ligand.symbols[i]),
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
      
    # Create ligand edges
    for i in range(n):
      for j in range(i + 1, n):
        distance = ligand.distance_matrix[i][j]
        bond = ligand.rdmol.GetBondBetweenAtoms(i, j)
        if bond is not None:
          ligand.graph.add_edge(i, j, is_intermolecular=False, is_covalent=True, bond_type=bond.GetBondType().name, distance=distance, interaction_type=[False, False, False])
        elif distance < self.dist_cutoff:
          ligand.graph.add_edge(i, j, is_intermolecular=False, is_covalent=False, bond_type="NON-COVALENT", distance=distance,
            interaction_type=judge_noncovalent_interaction_type(ligand.graph, i, j))

    pocket = SimpleNamespace()
    pocket.pdb_lines = receptor.pocket_pdb_block.split('\n')
    pocket.atom_idxs = []
    ATOM, resSeq_pre, resSeq, res_atom_idxs, coordinates = False, None, None, [], None
    for i, line in enumerate(pocket.pdb_lines):
      if (line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] != "HOH")) and line[76:78].strip() != 'H':
        ATOM = True
        resSeq = int(line[22:26])
        if resSeq != resSeq_pre:
          if resSeq_pre:
            min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
            if min_dist < self.dist_cutoff:
              pocket.atom_idxs.extend(res_atom_idxs)
          res_atom_idxs.append(i)
          coordinates = np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)
        else:
          res_atom_idxs.append(i)
          coordinates = np.concatenate((coordinates, np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)))
        resSeq_pre = resSeq
    if ATOM:
      min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
      if min_dist < self.dist_cutoff:
        pocket.atom_idxs.extend(res_atom_idxs)
    ligand.pocket_atom_idxs = pocket.atom_idxs
    pocket.positions = np.array([receptor.positions[i] for i in pocket.atom_idxs])
    ligand._pocket_distance_matrix = distance_matrix(ligand.positions, pocket.positions).tolist()

    return ligand

  def process(self):
    data_index_processed = []
    
    for i, index_row in self.data_index_df.iterrows():
      instance_name, ligand_name = index_row["instance_name"], index_row["ligand_name"]

      try:
        data = self.data_maker(index_row)
      except Exception as e:
        data = None
        logger.info(e)

      if data is None:
        logger.info(f"[{instance_name}] fail")
        continue

      if self.pre_filter is not None and not self.pre_filter(data):
        continue

      if self.pre_transform is not None:
        data = self.pre_transform(data)

      save_dir = os.path.join(self.processed_dir, instance_name, "ligands")
      pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
      with open(os.path.join(save_dir, f"{ligand_name}.pkl"), "wb") as f:
        pickle.dump(data, f)

      data_index_processed.append(list(index_row))

      logger.info(f"[{instance_name}/ligands/{ligand_name}] Success, Mol(num_atoms={data.rdmol.GetNumAtoms()})")
    
    data_index_processed_df = pd.DataFrame(data_index_processed, columns=self.data_index_df.columns)
    data_index_name, data_index_ext = os.path.splitext(self.data_index)
    data_index_processed_df.to_csv(os.path.join(self.root, data_index_name + ".processed" + data_index_ext), float_format='%.8f', index=False)

  def len(self):
    return len(self.data_index_df)

  def get(self, idx):
    index_row = self.data_index_df.iloc[idx]
    instance_name, ligand_name = index_row["instance_name"], index_row["ligand_name"]
    with open(os.path.join(self.processed_dir,  instance_name, "ligands", f"{ligand_name}.pkl"), "rb") as f:
      data = pickle.load(f)
    return data