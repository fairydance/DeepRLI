import os
import logging
import pickle
import pathlib
import numpy as np
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from scipy.spatial import distance_matrix
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from deeprli.data import Dataset
from deeprli.base import Attributes, ChemicalElements

logger = logging.getLogger(__name__)


class ReceptorDataset(Dataset):
  """Dataset class for processing receptor structures and binding site information.

  Handles both ligand-based and box-based binding site detection.
  """

  def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
               data_index="index/receptors.csv", ligand_file_types=["sdf"],
               dist_cutoff=6.5, bs_cutoff=6.5):
    """
    Args:
      root (str): Root directory for dataset storage
      data_index (str): Path to receptor index file
      ligand_file_types (list): Supported ligand file extensions
      dist_cutoff (float): Distance cutoff for edge creation (Å)
      bs_cutoff (float): Binding site detection cutoff (Å)
    """
    self.data_index = data_index
    self.ligand_file_types = ligand_file_types
    self.dist_cutoff = dist_cutoff
    self.bs_cutoff = bs_cutoff

    # Read index file with proper type conversion
    self.data_index_df = pd.read_csv(
      os.path.join(root, data_index),
      converters={
        "known_ligand_names": literal_eval,
        "box_center": lambda x: np.array(eval(x)) if x != 'nan' else None,
        "box_size": lambda x: np.array(eval(x)) if x != 'nan' else None
      }
    )

    # Van der Waals radii for atom types
    self.vdw_radii = {
      'C': 2.0, 'N': 1.7, 'O': 1.6, 'F': 1.5,
      'Si': 2.2, 'P': 2.1, 'S': 2.0, 'Cl': 1.8,
      'Br': 2.0, 'I': 2.2, 'At': 2.3,
      'Met': 1.2, 'Unk': 1.2
    }

    # Chemical feature factory initialization
    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    self.ptable = Chem.GetPeriodicTable()

    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
    """List of raw file paths based on instance names."""
    return [row["instance_name"] for _, row in self.data_index_df.iterrows()]

  @property
  def processed_file_names(self):
    """List of processed file paths."""
    return [f"{row['instance_name']}.pkl" for _, row in self.data_index_df.iterrows()]

  def modify_symbol(self, symbol, mode=0):
    """Normalize atom symbols to limited chemical types.

    Args:
      symbol (str): Original atom symbol
      mode (int): Aggressiveness of normalization
                  0: Basic metal detection
                  1: Advanced normalization
    """
    if symbol in ChemicalElements.metals:
      return 'Met'
    if mode != 0:
      if symbol == 'Se':
        return 'S'
      elif symbol not in ('C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I', 'At'):
        return 'Unk'
    return symbol

  def extract_features(self, molecule):
    """Extract chemical features from RDKit molecule."""
    molecule.symbols = [atom.GetSymbol() for atom in molecule.rdmol.GetAtoms()]
    molecule.positions = molecule.rdmol.GetConformer().GetPositions()

    # Calculate chemical features
    features = self.feature_factory.GetFeaturesForMol(molecule.rdmol)
    molecule.feature_dict = defaultdict(list)
    for feature in features:
      molecule.feature_dict[feature.GetFamily()].extend(feature.GetAtomIds())

  def process_residues(self, receptor_path, index_row):
    """Process receptor residues to identify binding site atoms.

    Uses either ligand positions or box parameters based on input availability.
    """
    # Load ligand positions if available
    ligand_positions = []
    if index_row["known_ligand_names"]:
      lig_dir = os.path.join(
        self.raw_dir, index_row["instance_name"], "ligands")
      for lig_name in index_row["known_ligand_names"]:
        for ext in self.ligand_file_types:
          lig_path = os.path.join(lig_dir, f"{lig_name}.{ext}")
          if os.path.exists(lig_path):
            if ext == "sdf":
              mol = Chem.SDMolSupplier(lig_path)[0]
            elif ext == "mol2":
              mol = Chem.MolFromMol2File(lig_path)
            elif ext == "pdb":
              mol = Chem.MolFromPDBFile(lig_path)
            if mol is not None:
              mol = Chem.RemoveHs(mol)
              ligand_positions.extend(mol.GetConformer().GetPositions())
              break

    # Determine processing method
    if index_row["box_center"] is not None and index_row["box_size"] is not None:
      return self._process_with_box(receptor_path, index_row)
    elif ligand_positions:
      return self._process_with_ligands(receptor_path, ligand_positions)
    else:
      logger.error(f"No binding site info for {index_row['instance_name']}")
      return None

  def _process_with_ligands(self, receptor_path, ligand_positions):
    """Identify residues near ligand atoms."""
    pocket_pdb_block = ""
    ligand_positions = np.array(ligand_positions)

    with open(receptor_path, "r") as f:
      current_res = []
      res_coords = []
      res_seq = None

      for line in f:
        if line.startswith("ATOM") or (line.startswith("HETATM")
                                       and line[17:20] != "HOH"):
          atom_coord = np.array([
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54])
          ])
          current_res.append(line)
          res_coords.append(atom_coord)

          # Check distance to ligands
          distances = np.linalg.norm(
            ligand_positions - atom_coord,
            axis=1
          )
          if np.any(distances < self.bs_cutoff):
            pocket_pdb_block += "".join(current_res)
            current_res = []
            res_coords = []

      # Handle last residue
      if res_coords:
        res_coords = np.array(res_coords)
        distances = np.linalg.norm(
          ligand_positions[:, None] - res_coords,
          axis=2
        )
        if np.any(distances < self.bs_cutoff):
          pocket_pdb_block += "".join(current_res)

    return pocket_pdb_block

  def _process_with_box(self, receptor_path, index_row):
    """Identify residues within specified box region."""
    box_center = np.array(index_row["box_center"])
    box_size = np.array(index_row["box_size"])
    half_size = box_size / 2
    bounds = np.array([
      box_center - half_size - self.bs_cutoff,
      box_center + half_size + self.bs_cutoff
    ])

    pocket_pdb_block = ""
    current_res = []

    with open(receptor_path, "r") as f:
      for line in f:
        if (line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] != "HOH")) and line[76:78].strip() != 'H':
          atom_coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
          # Check if atom is within expanded box
          if np.all(atom_coord >= bounds[0]) and np.all(atom_coord <= bounds[1]):
            current_res.append(line)
          else:
            if current_res:
              pocket_pdb_block += "".join(current_res)
              current_res = []

      # Add remaining residues
      if current_res:
        pocket_pdb_block += "".join(current_res)

    return pocket_pdb_block

  def data_maker(self, index_row):
    """Main processing pipeline for a receptor instance."""
    instance_name = index_row["instance_name"]
    receptor_path = os.path.join(
      self.raw_dir,
      instance_name,
      "receptor.pdb"
    )

    # Process binding site residues
    pocket_pdb_block = self.process_residues(receptor_path, index_row)
    if not pocket_pdb_block:
      logger.error(f"Failed to process binding site for {instance_name}")
      return None

    # Create receptor molecule
    receptor = Attributes()
    receptor.rdmol = Chem.MolFromPDBBlock(pocket_pdb_block)
    if receptor.rdmol is None:
      logger.error(f"Failed to parse receptor for {instance_name}")
      return None

    # Extract features and properties
    receptor.pocket_pdb_block = pocket_pdb_block
    receptor.rdmol = Chem.RemoveHs(receptor.rdmol)
    self.extract_features(receptor)
    receptor.distance_matrix = distance_matrix(
      receptor.positions,
      receptor.positions
    ).tolist()

    return receptor

  def process(self):
    """Batch processing of all receptor instances."""
    processed_data = []

    for _, row in self.data_index_df.iterrows():
      instance_name = row["instance_name"]
      try:
        receptor_data = self.data_maker(row)
        if receptor_data is None:
          continue

        if self.pre_filter and not self.pre_filter(receptor_data):
          continue

        if self.pre_transform:
          receptor_data = self.pre_transform(receptor_data)

        # Save processed data
        save_path = os.path.join(
          self.processed_dir,
          f"{instance_name}.pkl"
        )
        with open(save_path, "wb") as f:
          pickle.dump(receptor_data, f)

        processed_data.append(row)
        logger.info(
          f"Processed {instance_name} "
          f"(Atoms: {receptor_data.rdmol.GetNumAtoms()})"
        )

      except Exception as e:
        logger.error(f"Failed processing {instance_name}: {str(e)}")

    # Save processed index
    processed_df = pd.DataFrame(processed_data)
    index_name = os.path.splitext(self.data_index)[0]
    processed_df.to_csv(
      os.path.join(self.root, f"{index_name}.processed.csv"),
      index=False
    )

  def len(self):
    return len(self.data_index_df)

  def get(self, idx):
    instance_name = self.data_index_df.iloc[idx]["instance_name"]
    with open(os.path.join(self.processed_dir, f"{instance_name}.pkl"), "rb") as f:
      return pickle.load(f)
