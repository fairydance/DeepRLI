import os
import logging
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
from types import SimpleNamespace
from collections import defaultdict
from scipy.spatial import distance_matrix
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from pathlib import Path

from deeprli.data import Dataset

logger = logging.getLogger(__name__)

class ReceptorDataset(Dataset):
  """Dataset class for processing receptor structures with residue-level binding site detection.
  
  Features:
  - Processes PDB files residue-by-residue
  - Retains entire residue if any atom meets distance criteria
  - Supports both ligand-based and box-based binding site detection
  - Preserves original PDB formatting in output
  """

  def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
               data_index="index/receptors.csv", ligand_file_types=["sdf"],
               dist_cutoff=6.5, bs_cutoff=6.5):
    """
    Args:
      root (str): Root directory for dataset storage
      data_index (str): Path to receptor index file
      ligand_file_types (list): Supported ligand file extensions
      dist_cutoff (float): Distance cutoff for graph edge creation (Å)
      bs_cutoff (float): Binding site detection cutoff (Å). Use larger values (e.g., 8.0)
        for virtual screening with few known ligands to ensure complete residue inclusion.
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
      'C': 2.0, 'N': 1.7, 'O': 1.6, 'F': 1.5, 'Si': 2.2, 'P': 2.1, 'S': 2.0,
      'Cl': 1.8, 'Br': 2.0, 'I': 2.2, 'At': 2.3, 'Met': 1.2, 'Unk': 1.2
    }

    # Chemical feature factory initialization
    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    self.ptable = Chem.GetPeriodicTable()

    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
    """List of raw file paths based on instance names."""
    return [str(Path(row["instance_name"]/"receptor.pdb")) for _, row in self.data_index_df.iterrows()]

  @property
  def processed_file_names(self):
    """List of processed file paths."""
    return [str(Path(row["instance_name"]/"receptor.pkl")) for _, row in self.data_index_df.iterrows()]
  
  def _process_residues(self, receptor_path, index_row):
    """Process PDB file residue-by-residue with atomic distance checks.
    
    Returns:
      str: PDB block containing all residues with atoms within cutoff distance
    """
    # Load ligand positions if available
    ligand_coords = self._load_ligand_coordinates(index_row) if index_row["known_ligand_names"] else None
    
    # Calculate box boundaries if provided
    box_min, box_max = self._calculate_box_boundaries(index_row) if index_row["box_center"] is not None else (None, None)

    pocket_lines = []
    current_res = []
    current_coords = []
    res_seq = None

    with open(receptor_path, "r") as f:
      for line in f:
        if self._is_relevant_atom_line(line):
          atom_coord = self._parse_atom_coordinates(line)
          line_res_seq = int(line[22:26])

          if line_res_seq != res_seq:  # New residue
            if current_res:  # Process previous residue
              if self._residue_meets_criteria(current_coords, ligand_coords, box_min, box_max):
                pocket_lines.extend(current_res)
            # Start new residue
            current_res = [line]
            current_coords = [atom_coord]
            res_seq = line_res_seq
          else:  # Same residue
            current_res.append(line)
            current_coords.append(atom_coord)

      # Process final residue
      if current_res and self._residue_meets_criteria(current_coords, ligand_coords, box_min, box_max):
        pocket_lines.extend(current_res)

    return "".join(pocket_lines)
  
  def _is_relevant_atom_line(self, line):
    """Filter relevant atom lines (ATOM/HETATM, non-water, non-hydrogen)"""
    return (line.startswith("ATOM") or 
            (line.startswith("HETATM") and line[17:20] != "HOH")) and \
            line[76:78].strip() != 'H'

  def _parse_atom_coordinates(self, line):
    """Extract atomic coordinates from PDB line"""
    return np.array([
      float(line[30:38]),  # x
      float(line[38:46]),  # y
      float(line[46:54])   # z
    ])

  def _load_ligand_coordinates(self, index_row):
    """Load coordinates from all known ligands"""
    coords = []
    instance_dir = Path(self.raw_dir)/index_row["instance_name"]
    
    for lig_name in index_row["known_ligand_names"]:
      for ext in self.ligand_file_types:
        lig_path = instance_dir/"ligands"/f"{lig_name}.{ext}"
        if lig_path.exists():
          if ext == "sdf":
            mol = Chem.SDMolSupplier(lig_path)[0]
          elif ext == "mol2":
            mol = Chem.MolFromMol2File(lig_path)
          elif ext == "pdb":
            mol = Chem.MolFromPDBFile(lig_path)
          if mol:
            mol = Chem.RemoveHs(mol)
            coords.extend(mol.GetConformer().GetPositions())
            break
    return np.array(coords) if coords else None

  def _calculate_box_boundaries(self, index_row):
    """Calculate expanded box boundaries including cutoff"""
    center = index_row["box_center"]
    size = index_row["box_size"]
    half_size = size/2 + self.bs_cutoff
    return center - half_size, center + half_size

  def _residue_meets_criteria(self, residue_coords, ligand_coords, box_min, box_max):
    """Check if any atom in residue meets inclusion criteria"""
    # Priority: box > ligands
    if box_min is not None:
      return np.any((residue_coords > box_min) & (residue_coords < box_max))
    if ligand_coords is not None:
      distances = np.linalg.norm(
        ligand_coords[:, np.newaxis] - residue_coords,
        axis=2
      )
      return np.any(distances < self.bs_cutoff)
    return False
  
  def _process_receptor(self, instance_name, pocket_pdb):
    """Create receptor object from PDB block"""
    receptor = SimpleNamespace()
    receptor.rdmol = Chem.MolFromPDBBlock(pocket_pdb)
    if receptor.rdmol is None:
      raise ValueError(f"Failed to parse receptor {instance_name}")
    
    # Process receptor properties
    receptor.pocket_pdb_block = pocket_pdb
    receptor.rdmol = Chem.RemoveHs(receptor.rdmol)
    receptor.symbols = [atom.GetSymbol() for atom in receptor.rdmol.GetAtoms()]
    receptor.positions = receptor.rdmol.GetConformer().GetPositions()
    self._extract_features(receptor)
    receptor.distance_matrix = distance_matrix(receptor.positions, receptor.positions).tolist()
    return receptor

  def _extract_features(self, molecule):
    """Extract chemical features using RDKit feature factory"""
    features = self.feature_factory.GetFeaturesForMol(molecule.rdmol)
    molecule.feature_dict = defaultdict(list)
    for feat in features:
      molecule.feature_dict[feat.GetFamily()].extend(feat.GetAtomIds())

  def process(self):
    """Main processing pipeline"""
    processed_index_rows = []
    
    for _, index_row in self.data_index_df.iterrows():
      instance_name = index_row["instance_name"]
      try:
        # Validate input
        receptor_path = Path(self.raw_dir)/instance_name/"receptor.pdb"
        if not receptor_path.exists():
          logger.error(f"Receptor file missing: {receptor_path}")
          continue

        # Process binding site
        pocket_pdb = self._process_residues(str(receptor_path), index_row)
        if not pocket_pdb:
          logger.warning(f"No binding site detected for {instance_name}")
          continue

        # Create receptor object
        receptor = self._process_receptor(instance_name, pocket_pdb)
        
        # Apply filters/transforms
        if self.pre_filter and not self.pre_filter(receptor):
          continue
        if self.pre_transform:
          receptor = self.pre_transform(receptor)

        # Save results
        save_dir = Path(self.processed_dir)/instance_name
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = save_dir/"receptor.pkl"
        with open(save_path, "wb") as f:
          pickle.dump(receptor, f)
        
        processed_index_rows.append(list(index_row))
        logger.info(f"Processed {instance_name} ({len(receptor.positions)} atoms)")

      except Exception as e:
        logger.error(f"Failed processing {instance_name}: {str(e)}")

    # Save processing metadata
    data_index_name, data_index_ext = os.path.splitext(self.data_index)
    pd.DataFrame(processed_index_rows, columns=self.data_index_df.columns).to_csv(
      Path(self.root)/f"{data_index_name}.processed{data_index_ext}",
      float_format='%.8f', index=False
    )

  def len(self):
    return len(self.data_index_df)

  def get(self, idx):
    instance_name = self.data_index_df.iloc[idx]["instance_name"]
    with open(Path(self.processed_dir)/instance_name/"receptor.pkl", "rb") as f:
      return pickle.load(f)
