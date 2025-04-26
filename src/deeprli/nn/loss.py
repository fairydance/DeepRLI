import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, result, affinity):
    return torch.mean((result - affinity)**2)


class ContrastiveLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, result, affinity):
    anchor_affinity_difference = result[0][0] - affinity[0]
    negative_redock_anchor_difference = result[1][1] - result[2][1]
    negative_crossdock_anchor_difference = result[1][2] - result[3][2]
    random_affinity_difference = result[4][0] - affinity[1]

    anchor_affinity_loss = anchor_affinity_difference**2
    negative_redock_anchor_loss = F.relu(negative_redock_anchor_difference)
    negative_crossdock_anchor_loss = F.relu(negative_crossdock_anchor_difference)
    random_affinity_loss = random_affinity_difference**2

    return torch.mean(anchor_affinity_loss + negative_redock_anchor_loss + negative_crossdock_anchor_loss + random_affinity_loss)