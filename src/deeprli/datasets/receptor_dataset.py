import os, pickle, logging, pathlib
from ast import literal_eval
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from deeprli.data import Dataset
from deeprli.base import Attributes, ChemicalElements

logger = logging.getLogger(__name__)

class ReceptorDataset(Dataset):
  '''Receptor Dataset'''
  def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
               data_index="index/data.csv", ligand_file_types=["sdf", "mol2"], dist_cutoff=6.5, bs_cutoff=6.5,
               complex_path_prefix=""):
    self.data_index = data_index
    self.ligand_file_types = ligand_file_types
    self.dist_cutoff = dist_cutoff
    self.bs_cutoff = bs_cutoff
    self.complex_path_prefix = complex_path_prefix

    self.data_index_df = pd.read_csv(os.path.join(root, data_index), converters={"known_ligands": literal_eval})
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
  
  def data_maker(self, index_row):
    complex_path = index_row["complex_path"]
    known_ligands = index_row["known_ligands"]

    ## Parse Ligand
    ligands = []
    for known_ligand in known_ligands:
      ligand = Attributes({"rdmol": None})
      for ligand_file_type in self.ligand_file_types:
        ligand_file_path = os.path.join(self.raw_dir, self.complex_path_prefix, complex_path, "ligands", f"{known_ligand}.{ligand_file_type}")
        if not os.path.exists(ligand_file_path):
          logger.info(f"[{complex_path}] Ligand File Not Found (.{ligand_file_type}): {ligand_file_path}")
        else:
          if ligand_file_type.split('.')[-1] == "sdf":
            ligand = Attributes({"rdmol": Chem.SDMolSupplier(ligand_file_path)[0]})
          elif ligand_file_type.split('.')[-1] == "mol2":
            ligand = Attributes({"rdmol": Chem.MolFromMol2File(ligand_file_path)})
          elif ligand_file_type.split('.')[-1] == "pdb":
            ligand = Attributes({"rdmol": Chem.MolFromPDBFile(ligand_file_path)})
          if ligand.rdmol is None:
            logger.info(f"[{complex_path}] Ligand Molecule Parsing Failed (.{ligand_file_type}): {ligand_file_path}")
          else:
            break

      if ligand.rdmol is None:
        return None

      ligand.rdmol = Chem.RemoveHs(ligand.rdmol)
      self.extract_features(ligand)
      ligands.append(ligand)
    
    ## Parse Receptor
    receptor_file_path = os.path.join(self.raw_dir, self.complex_path_prefix, complex_path, "receptor.pdb")
    if not os.path.exists(receptor_file_path):
      logger.info(f"[{complex_path}] Receptor File Not Found (.pdb): {receptor_file_path}")
      return None

    pocket_pdb_block = ""
    with open(receptor_file_path, "rt") as f:
      ATOM, resSeq_pre, resSeq, res_lines, coordinates = False, None, None, None, None
      for line in f:
        if (line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] != "HOH")) and line[76:78].strip() != 'H':
          ATOM = True
          resSeq = int(line[22:26])
          if resSeq != resSeq_pre:
            if resSeq_pre:
              for ligand in ligands:
                min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
                if min_dist < self.bs_cutoff:
                  pocket_pdb_block += res_lines
                  break
            res_lines = line
            coordinates = np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)
          else:
            res_lines += line
            coordinates = np.concatenate((coordinates, np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)))
          resSeq_pre = resSeq
        # if line.startswith("CONECT"):
        #   pocket_pdb_block += line
      if ATOM:
        for ligand in ligands:
          min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
          if min_dist < self.bs_cutoff:
            pocket_pdb_block += res_lines
            break

    receptor = Attributes({"rdmol": Chem.MolFromPDBBlock(pocket_pdb_block)})

    if receptor.rdmol is None:
      logger.info(f"[{complex_path}] Receptor Molecule Parsing Failed (.pdb): {receptor_file_path}")
      return None
    
    receptor.pocket_pdb_block = pocket_pdb_block
    receptor.rdmol = Chem.RemoveHs(receptor.rdmol)
    self.extract_features(receptor)

    receptor.distance_matrix = distance_matrix(receptor.positions, receptor.positions).tolist()

    return receptor

  def process(self):
    data_index_processed = []
    
    for i, index_row in self.data_index_df.iterrows():
      complex_path = index_row["complex_path"]
      complex_dir = complex_path
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

      pathlib.Path(os.path.join(self.processed_dir, complex_dir)).mkdir(parents=True, exist_ok=True)
      with open(os.path.join(self.processed_dir, complex_dir, "receptor.pkl"), "wb") as f:
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
