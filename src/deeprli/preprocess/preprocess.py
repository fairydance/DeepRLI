import sys, time, argparse, logging

from deeprli.datasets import ComplexDataset

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Data Preprocessing")
  parser.add_argument("--data-root", type=str, default="dataset", help="Root directory where the dataset should be saved")
  parser.add_argument("--data-index", type=str, default="index/data.csv", help="Data index file path relative to root")
  parser.add_argument("--ligand-file-types", type=str, default="sdf,mol2,mol2.obabel.sdf,mol2.obabel.pdb", help="ligand file types separated by commas")
  parser.add_argument("--dist-cutoff", type=float, default=6.5, help="distance cutoff for residues around the ligand")
  parser.add_argument("--save-details", action="store_true", help="save details of ligand and receptor or not")
  parser.add_argument("--save-single", action="store_true", help="save every single processed data")
  args = parser.parse_args()

  logging.root.handlers = []
  logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
      logging.StreamHandler(sys.stdout)
    ]
  )

  logger = logging.getLogger(__name__)
  logger.info("Preprocessing Records")

  now = time.localtime()
  timestamp = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
  logger.info(f"Date: {timestamp}")
  logger.info(args)

  ComplexDataset(
    root=args.data_root,
    data_index=args.data_index,
    ligand_file_types=args.ligand_file_types.split(','),
    dist_cutoff=args.dist_cutoff,
    save_details=args.save_details,
    save_single=args.save_single
  )

  logger.info("Preprocess complete. Congratulations!")