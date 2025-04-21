import os, sys, argparse, logging, shutil
from pathlib import Path
import pandas as pd
from rdkit import Chem

from deeprli.datasets import ReceptorDataset

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Preprocess Receptor Data")
  input_group = parser.add_mutually_exclusive_group(required=True)
  input_group.add_argument("--data-index", type=str,
                           help="Path to existing data index file (relative to data-root)")
  input_group.add_argument("--receptor-file-path", type=str,
                           help="Path to receptor PDB file for new instance")

  parser.add_argument("--data-root", type=str, default="dataset",
                      help="Root directory for dataset storage")
  parser.add_argument("--instance-name", type=str,
                      help="Unique identifier for the receptor instance")
  parser.add_argument("--known-ligand-file-path", type=str,
                      help="Path to SDF file containing multiple known ligands")
  parser.add_argument("--box-center", nargs=3, type=float,
                      help="Binding box center coordinates [x,y,z]")
  parser.add_argument("--box-size", nargs=3, type=float,
                      help="Binding box dimensions [x_size,y_size,z_size]")
  parser.add_argument("--ligand-file-types", type=str, default="sdf",
                      help="Comma-separated ligand file extensions")
  parser.add_argument("--dist-cutoff", type=float, default=6.5,
                      help="Distance cutoff for edge creation (Å)")
  parser.add_argument("--bs-cutoff", type=float, default=6.5,
                      help="Distance cutoff (Å) for binding site detection. " 
                           "In virtual screening scenarios with few known ligands, " 
                           "consider using larger values (e.g., 8.0) to ensure "
                           "sufficient binding site coverage.")
  args = parser.parse_args()

  # Configure logging
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
  )
  logger = logging.getLogger(__name__)
  logger.info("Starting receptor preprocessing")

  # Handle new receptor instance creation
  if not args.data_index:
    # Validate required parameters
    if not args.receptor_file_path:
      logger.error("--receptor-file-path required when not using --data-index")
      sys.exit(1)

    # Create instance directory structure
    data_root = Path(args.data_root)
    instance_name = args.instance_name or Path(args.receptor_file_path).stem
    instance_dir = data_root/"raw"/instance_name
    ligand_dir = instance_dir/"ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)

    # Copy receptor file
    shutil.copy(args.receptor_file_path, instance_dir/"receptor.pdb")
    logger.info(f"Created receptor instance: {instance_name}")

    # Process multi-ligand SDF file if provided
    known_ligands = []
    if args.known_ligand_file_path:
      if not os.path.exists(args.known_ligand_file_path):
        logger.error(f"Ligand file not found: {args.known_ligand_file_path}")
        sys.exit(1)

      # Extract individual ligands from SDF
      suppl = Chem.SDMolSupplier(args.known_ligand_file_path)
      for idx, mol in enumerate(suppl):
        if mol is None:
          logger.warning(f"Skipping invalid molecule at position {idx}")
          continue

        # Generate ligand name
        lig_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"ligand_{idx+1}"
        output_path = ligand_dir/f"{lig_name}.sdf"

        # Write single ligand SDF
        with Chem.SDWriter(str(output_path)) as writer:
          writer.write(mol)
        known_ligands.append(lig_name)
        logger.info(f"Extracted ligand: {lig_name}")

    # Create receptor index entry
    index_entry = {
      "instance_name": [instance_name],
      "known_ligand_names": [known_ligands],
      "box_center": [args.box_center],
      "box_size": [args.box_size]
    }

    # Save to index file
    index_path = data_root/"index"/"receptors.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Write new or append to existing
    if index_path.exists():
      index_df = pd.read_csv(index_path)
      index_df = pd.concat([index_df, pd.DataFrame(index_entry)])
    else:
      index_df = pd.DataFrame(index_entry)

    index_df.to_csv(index_path, index=False)
    args.data_index = "index/receptors.csv"
    logger.info(f"Updated receptor index at {index_path}")

  # Initialize dataset processing
  ReceptorDataset(
    root=args.data_root,
    data_index=args.data_index,
    ligand_file_types=args.ligand_file_types.split(','),
    dist_cutoff=args.dist_cutoff,
    bs_cutoff=args.bs_cutoff
  )
  logger.info("Receptor preprocessing completed successfully")
