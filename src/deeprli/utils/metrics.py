import numpy as np
from scipy import stats

def mae(p, y):
  """Mean Absolute Error"""
  return np.mean(np.absolute(p - y))

def mse(p, y):
  """Mean Square Error"""
  return np.mean((p - y)**2)

def rmse(p, y):
  """Root Mean Square Error"""
  return np.sqrt(np.mean((p - y)**2))

def pearson(p, y):
  """Pearson correlation coefficient"""
  return np.corrcoef(p, y)[0, 1]

def spearman(p, y):
  """Spearman correlation coefficient"""
  return stats.spearmanr(p, y)[0]

def ci(p, y):
  """Concordance Index"""
  ind = np.argsort(y)
  p, y = p[ind], y[ind]
  s, z = 0., 0.
  for i in range(1, len(y)):
    for j in range(i):
      if y[i] > y[j]:
        z += 1
        u = p[i] - p[j]
        if u > 0:
          s += 1
        elif u == 0:
          s += 0.5
  ci = s / z
  return ci