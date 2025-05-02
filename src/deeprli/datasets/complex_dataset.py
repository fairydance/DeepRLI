import os, pickle, logging, pathlib, time
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors
import networkx as nx
import dgl

from deeprli.data import Dataset
from deeprli.base import ChemicalElements
from deeprli.utils import one_hot_encoding, judge_noncovalent_interaction_type

logger = logging.getLogger(__name__)

class ComplexDataset(Dataset):
  def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
               data_index="index/data.csv", ligand_file_types=["sdf", "mol2", "pdb"], dist_cutoff=6.5,
               save_details=False, save_single=False):
    self.data_index = data_index
    self.ligand_file_types = ligand_file_types
    self.dist_cutoff = dist_cutoff
    self.save_details = save_details
    self.save_single = save_single

    self.data_index_df = pd.read_csv(os.path.join(root, data_index))
    self.vdw_radii = {'C': 2.0, 'N': 1.7, 'O': 1.6, 'F': 1.5, 'Si': 2.2, 'P': 2.1, 'S': 2.0, 'Cl': 1.8, 'Br': 2.0,
                      'I': 2.2, 'At': 2.3, 'Met': 1.2, 'Unk': 1.2}
    
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    self.ptable = Chem.GetPeriodicTable()

    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def compiled_dir(self):
    return os.path.join(self.root, "compiled")

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
  
  def data_maker(self, complex_path):
    ## Parse Ligand
    ligand = SimpleNamespace(rdmol=None)
    for ligand_file_type in self.ligand_file_types:
      ligand_file_path = os.path.join(self.raw_dir, complex_path, f"ligand.{ligand_file_type}")
      if not os.path.exists(ligand_file_path):
        logger.info(f"[{complex_path}] Ligand File Not Found (.{ligand_file_type}): {ligand_file_path}")
      else:
        if ligand_file_type.split('.')[-1] == "sdf":
          ligand.rdmol = Chem.SDMolSupplier(ligand_file_path)[0]
        elif ligand_file_type.split('.')[-1] == "mol2":
          ligand.rdmol = Chem.MolFromMol2File(ligand_file_path)
        elif ligand_file_type.split('.')[-1] == "pdb":
          ligand.rdmol = Chem.MolFromPDBFile(ligand_file_path)
        if ligand.rdmol is None:
          logger.info(f"[{complex_path}] Ligand Molecule Parsing Failed (.{ligand_file_type}): {ligand_file_path}")
        else:
          break

    if ligand.rdmol is None:
      return None

    ligand.rdmol = Chem.RemoveHs(ligand.rdmol)
    self.extract_features(ligand)

    ligand.num_rotatable_bonds = Descriptors.NumRotatableBonds(ligand.rdmol)
    
    ## Parse Receptor
    receptor_file_path = os.path.join(self.raw_dir, complex_path, "receptor.pdb")
    pocket_file_path = os.path.join(self.raw_dir, complex_path, f"pocket_{self.dist_cutoff:.2f}A.pdb")
    if os.path.isfile(pocket_file_path):
      with open(pocket_file_path, "rt") as f:
        pocket_pdb_block = f.read()
    else:
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
                min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
                if min_dist < self.dist_cutoff:
                  pocket_pdb_block += res_lines
              res_lines = line
              coordinates = np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)
            else:
              res_lines += line
              coordinates = np.concatenate((coordinates, np.array([[line[30:38], line[38:46], line[46:54]]], dtype=float)))
            resSeq_pre = resSeq
        if ATOM:
          min_dist = np.linalg.norm(coordinates[:, np.newaxis] - ligand.positions, axis=2).min()
          if min_dist < self.dist_cutoff:
            pocket_pdb_block += res_lines

      # with open(pocket_file_path, "wt") as f:
      #   f.write(pocket_pdb_block)

    receptor = SimpleNamespace(rdmol=Chem.MolFromPDBBlock(pocket_pdb_block))

    if receptor.rdmol is None:
      logger.info(f"[{complex_path}] Receptor Molecule Parsing Failed (.pdb): {receptor_file_path}")
      return None
    
    receptor.pocket_pdb_block = pocket_pdb_block
    receptor.rdmol = Chem.RemoveHs(receptor.rdmol)
    self.extract_features(receptor)

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
    receptor_receptor_distance_matrix = distance_matrix(receptor.positions, receptor.positions).tolist()
    ligand_receptor_distance_matrix = distance_matrix(ligand.positions, receptor.positions).tolist()

    ## Create Ligand Edges
    for i in range(n1):
      for j in range(i + 1, n1):
        distance = ligand_ligand_distance_matrix[i][j]
        bond = ligand.rdmol.GetBondBetweenAtoms(i, j)
        if bond is not None:
          g.add_edge(i, j, is_intermolecular=False, is_covalent=True, bond_type=bond.GetBondType().name, distance=distance, interaction_type=[False, False, False])
        elif distance < self.dist_cutoff:
          g.add_edge(i, j, is_intermolecular=False, is_covalent=False, bond_type="NON-COVALENT", distance=distance,
            interaction_type=judge_noncovalent_interaction_type(g, i, j))

    ## Create Receptor Edges
    for i in range(n2):
      for j in range(i + 1, n2):
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
        distance = ligand_receptor_distance_matrix[i][j]
        if distance < self.dist_cutoff:
          g.add_edge(i, j + n1, is_intermolecular=True, is_covalent=False, bond_type="NON-COVALENT", distance=distance,
            interaction_type=judge_noncovalent_interaction_type(g, i, j + n1))

    complex = SimpleNamespace()
    self.get_node_features(g)
    self.get_edge_features(g)
    complex.graph = dgl.from_networkx(g, node_attrs=["feature", "vdw_radii"], edge_attrs=["feature", "distance", "interaction_type"])

    data = {
      "complex.graph": complex.graph,
      "ligand.num_atoms": n1,
      "ligand.num_rotatable_bonds": ligand.num_rotatable_bonds,
      "receptor.num_atoms": n2,
    }

    if self.save_details:
      data["ligand"] = ligand
      data["receptor"] = receptor

    return data

  def process(self):
    all_processed_data = {}
    data_index_processed = []
    start = time.perf_counter()
    for i, index_row in self.data_index_df.iterrows():
      complex_path = index_row["complex_path"]
      complex_dir, complex_name = complex_path.rsplit('/', 1) if '/' in complex_path else ('', complex_path)
      data = self.data_maker(complex_path)

      if data is None:
        continue

      if self.pre_filter is not None and not self.pre_filter(data):
        continue

      if self.pre_transform is not None:
        data = self.pre_transform(data)

      if self.save_single:
        pathlib.Path(os.path.join(self.processed_dir, complex_dir)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.processed_dir, f"{complex_path}.pkl"), "wb") as f:
          pickle.dump(data, f)

      all_processed_data[complex_path] = data
      data_index_processed.append(list(index_row))

      logger.info(f"[{index_row['complex_path']}] Success, Graph(num_nodes={data['complex.graph'].num_nodes()}, num_edges={data['complex.graph'].num_edges()})")
    end = time.perf_counter()
    duration = end - start
    avg_per_item = duration / len(self.data_index_df)

    logger.info(f"Data processing complete. Time of duration: {duration:.6f} seconds for {len(self.data_index_df)} item(s). Average per item: {avg_per_item:.6f} seconds/item")

    pathlib.Path(self.compiled_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(self.compiled_dir, f"{os.path.splitext(os.path.basename(self.data_index))[0]}.processed.pkl"), "wb") as f:
      pickle.dump(all_processed_data, f)

    data_index_processed_df = pd.DataFrame(data_index_processed, columns=self.data_index_df.columns)
    data_index_name, data_index_ext = os.path.splitext(self.data_index)
    data_index_processed_df.to_csv(os.path.join(self.root, data_index_name + ".processed" + data_index_ext), float_format="%.8f", index=False)
