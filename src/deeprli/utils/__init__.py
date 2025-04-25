import random
import numpy as np
import torch
import torch.backends.cudnn
import torch.linalg as LA

from deeprli.base import ChemicalElements
from .dict_queue import DictQueue
from .makedirs import makedirs


def set_deterministic(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def is_hydrophobic_interaction(g, i, j):
  def is_hydrophobe(g, i):
    return (g.nodes[i]["symbol"] == 'C' and (g.nodes[i]["is_hydrophobe"] or g.nodes[i]["is_lumped_hydrophobe"])) or (g.nodes[i]["symbol"] in ChemicalElements.halogens)
    
  return is_hydrophobe(g, i) and is_hydrophobe(g, j)

def is_hydrogen_bond(g, i, j):
  def is_donor(g, i):
    return (g.nodes[i]["symbol"] in ('N', 'O') and g.nodes[i]["is_donor"]) or (g.nodes[i]["symbol"] == "Met")
    
  def is_acceptor(g, i):
    return g.nodes[i]["symbol"] in ('N', 'O') and g.nodes[i]["is_acceptor"]
    
  return (is_donor(g, i) and is_acceptor(g, j)) or (is_acceptor(g, i) and is_donor(g, j))

def judge_noncovalent_interaction_type(g, i, j):
  if is_hydrophobic_interaction(g, i, j):
    return [False, True, False]
  elif is_hydrogen_bond(g, i, j):
    return [False, False, True]
  else:
    return [True, False, False]
  

def calc_centroidal_dist():
  def func(edges):
    return {'d': LA.vector_norm(edges.dst['x'] - edges.src['x'], dim=1)}
  return func

def calc_reduced_dist():
  def func(edges):
    return {'rd': edges.data['d'] - edges.src['r'] - edges.dst['r']}
  return func

def one_hot_encoding(x, allowable_set, with_unknown=False):
  if with_unknown and (x not in allowable_set):
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))
