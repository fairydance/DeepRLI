import argparse
import torch

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Export state dict from model object")
  parser.add_argument("--model", type=str, help="the path of a trained model")
  parser.add_argument("--save-path", type=str, help="the path to deposit state_dict")
  args = parser.parse_args()

  device = torch.device("cpu")
  model = torch.load(args.model, map_location=device)
  torch.save(model.state_dict(), args.save_path)
