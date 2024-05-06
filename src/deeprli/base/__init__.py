from .chemical_constants import ChemicalConstants
from .chemical_elements import ChemicalElements


class Attributes(dict):
  def __getattr__(self, key):
    if key not in self:
      return dict.__getattribute__(self, key)
    return self[key]

  def __setattr__(self, key, value):
    self[key] = value