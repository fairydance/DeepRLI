import os, pickle, logging, pathlib, json, gzip
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors
import networkx as nx
import dgl
import torch

from deeprli.data import Dataset
from deeprli.base import Attributes, ChemicalElements
from deeprli.utils import one_hot_encoding

logger = logging.getLogger(__name__)

class LigandDataset(Dataset):
  '''Ligand Dataset'''
  def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
               data_index="index/data.csv", ligand_file_types=["mol2", "sdf"], dist_cutoff=6.5):
    self.data_index = data_index
    self.ligand_file_types = ligand_file_types
    self.dist_cutoff = dist_cutoff

    self.data_index_df = pd.read_csv(os.path.join(root, data_index))
    self.vdw_radii = {'C': 2.0, 'N': 1.7, 'O': 1.6, 'F': 1.5, 'Si': 2.2, 'P': 2.1, 'S': 2.0, 'Cl': 1.8, 'Br': 2.0,
                      'I': 2.2, 'At': 2.3, 'Met': 1.2, 'Unk': 1.2}
    
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    self.ptable = Chem.GetPeriodicTable()

    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
    return [complex_path for complex_path in self.data_index_df["complex_path"]]

  @property
  def processed_file_names(self):
    return [f"{complex_path}.pkl" for complex_path in self.data_index_df["complex_path"]]
  
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

  def gaussian_smearing(self, values, start, end, steps):
    offsets = np.linspace(start, end, steps)
    width = offsets[1] - offsets[0]
    return np.exp(-0.5 / np.power(width, 2) * np.power(values[..., None] - offsets[None, ...], 2))
  
  def bessel_smearing(self, values, cutoff, num):
    return np.sqrt(2 / cutoff) * np.sin((np.arange(num) + 1) * np.pi * values[..., None] / cutoff) / values[..., None]

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
  
  def data_maker(self, index_row):
    complex_path = index_row["complex_path"]
    complex_dir, ligand_name = complex_path.rsplit('/', 1)
    
    ## Parse Ligand
    ligand = Attributes({"rdmol": None})
    for ligand_file_type in self.ligand_file_types:
      ligand_file_path = os.path.join(self.raw_dir, complex_dir, "ligands", f"{ligand_name}.{ligand_file_type}")
      if not os.path.exists(ligand_file_path):
        logger.info(f"[{complex_dir}] Ligand File Not Found (.{ligand_file_type}): {ligand_file_path}")
      else:
        if ligand_file_type.split('.')[-1] == "sdf":
          ligand = Attributes({"rdmol": Chem.SDMolSupplier(ligand_file_path)[0]})
        elif ligand_file_type.split('.')[-1] == "mol2":
          ligand = Attributes({"rdmol": Chem.MolFromMol2File(ligand_file_path)})
        elif ligand_file_type.split('.')[-1] == "pdb":
          ligand = Attributes({"rdmol": Chem.MolFromPDBFile(ligand_file_path)})
        if ligand.rdmol is None:
          logger.info(f"[{complex_dir}] Ligand Molecule Parsing Failed (.{ligand_file_type}): {ligand_file_path}")
        else:
          break

    if ligand.rdmol is None:
      return None

    ligand.rdmol = Chem.RemoveHs(ligand.rdmol)
    self.extract_features(ligand)

    ligand.num_rotatable_bonds = Descriptors.NumRotatableBonds(ligand.rdmol)

    return ligand

  def process(self):
    data_index_processed = []
    
    for i, index_row in self.data_index_df.iterrows():
      complex_path = index_row["complex_path"]
      complex_dir, ligand_name = complex_path.rsplit('/', 1)
      try:
        data = self.data_maker(index_row)
      except Exception as e:
        data = None
        logger.info(e)

      if data is None:
        logger.info(f"[{complex_path}] fail")
        continue

      if self.pre_filter is not None and not self.pre_filter(data):
        continue

      if self.pre_transform is not None:
        data = self.pre_transform(data)

      save_dir = os.path.join(self.processed_dir, complex_dir, "ligands")
      pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
      with open(os.path.join(save_dir, f"{ligand_name}.pkl"), "wb") as f:
        pickle.dump(data, f)

      data_index_processed.append(list(index_row))

      logger.info(f"[{index_row['complex_path']}] Success, Mol(num_atoms={data.rdmol.GetNumAtoms()})")
    
    data_index_processed_df = pd.DataFrame(data_index_processed, columns=self.data_index_df.columns)
    data_index_name, data_index_ext = os.path.splitext(self.data_index)
    data_index_processed_df.to_csv(os.path.join(self.root, data_index_name + ".processed" + data_index_ext), float_format='%.8f', index=False)

  def len(self):
    return len(self.data_index_df)

  def get(self, idx):
    complex_path = self.data_index_df["complex_path"][idx]
    with open(os.path.join(self.processed_dir, f"{complex_path}.pkl"), "rb") as f:
      data = pickle.load(f)
    return data