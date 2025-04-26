import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .modules import GraphTransformerEdgeLayer, MLPBlock


def calc_reduced_dist(edges):
  return {"reduced_distance": edges.data["distance"] - edges.src["vdw_radii"] - edges.dst["vdw_radii"]}

def u_v_add(edges):
  return {'x': edges.src['v'] + edges.dst['v']}


class DeepRLI(torch.nn.Module):
  def __init__(
    self,
    f_dropout_rate,
    g_dropout_rate,
    hidden_dim,
    num_attention_heads,
    use_layer_norm,
    use_batch_norm,
    use_residual,
    use_envelope=True,
    use_multi_obj=True
  ):
    super().__init__()
    self.__name__ = "DeepRLI"
    self.use_multi_obj = use_multi_obj
    self.embedding_v = torch.nn.Linear(39, hidden_dim, bias=True)
    self.embedding_e = torch.nn.Linear(39, hidden_dim, bias=True)

    self.f_feat_dropout = torch.nn.Dropout(f_dropout_rate)

    self.graph_layers = torch.nn.ModuleList([GraphTransformerEdgeLayer(hidden_dim, hidden_dim, num_attention_heads, g_dropout_rate,
                        use_layer_norm, use_batch_norm, use_residual, use_envelope) for _ in range(10)])
    
    self.readout1 = MLPBlock(hidden_dim, 1)
    if self.use_multi_obj:
      self.readout2 = MLPBlock(hidden_dim, 4)
      self.readout3 = MLPBlock(hidden_dim, 4)
      self.readout4 = MLPBlock(hidden_dim, 1)

  def V_steric_attraction(self, d):
    return -0.045 * torch.exp(-(d / 0.8)**2)
  
  def V_steric_repulsion(self, d):
    return 0.8 * (d < 0) * d**2
  
  def V_hydrophobic(self, d):
    return -0.035 * ((d > 0) * (d < 2.5) * (-0.4 * (d - 2.5)) + (d <= 0) * torch.ones_like(d))
  
  def V_hbond(self, d):
    return -0.6 * ((d > -0.6) * (d < 0) * (-5/3 * d) + (d <= -0.6) * torch.ones_like(d))

  def forward(self, inputs):
    g = inputs["complex.graph"]
    N_rot = inputs["ligand.num_rotatable_bonds"]
    g.apply_edges(calc_reduced_dist)
    
    v = self.embedding_v(g.ndata["feature"])
    v = self.f_feat_dropout(v)
    e = self.embedding_e(g.edata["feature"])
    for graph_layer in self.graph_layers:
      v, e = graph_layer(g, v, e)

    v = v * g.ndata["feature"][:, 0][..., None]
    g.ndata['v'] = v

    z = dgl.sum_nodes(g, 'v')
    y1 = self.readout1(z).squeeze(-1)

    if not self.use_multi_obj:
      return y1

    g.apply_edges(u_v_add)
    V = [
      self.V_steric_attraction(g.edata["reduced_distance"]),
      self.V_steric_repulsion(g.edata["reduced_distance"]),
      self.V_hydrophobic(g.edata["reduced_distance"]),
      self.V_hbond(g.edata["reduced_distance"])
    ]

    w2 = F.sigmoid(self.readout2(g.edata['x'])) + 0.5
    y2 = (g.edata["distance"] < 6.5) * g.edata["feature"][:, 0] * (
      w2[:, 0] * V[0] + w2[:, 1] * V[1] +
      g.edata["interaction_type"][:, 1] * w2[:, 2] * V[2] +
      g.edata["interaction_type"][:, 2] * w2[:, 3] * V[3]
    )
    g.edata['y2'] = y2
    y2 = dgl.sum_edges(g, 'y2') / 2

    w3 = F.sigmoid(self.readout3(g.edata['x'])) + 0.5
    y3 = (g.edata["distance"] < 6.5) * g.edata["feature"][:, 0] * (
      w3[:, 0] * V[0] + w3[:, 1] * V[1] +
      g.edata["interaction_type"][:, 1] * w3[:, 2] * V[2] +
      g.edata["interaction_type"][:, 2] * w3[:, 3] * V[3]
    )
    g.edata['y3'] = y3
    y3 = dgl.sum_edges(g, 'y3') / 2
    y3 /= 1 + (F.sigmoid(self.readout4(z)).squeeze(-1) + 0.5) * 0.05846 * N_rot

    return y1, y2, y3


class DeepRLIForInterpretation(torch.nn.Module):
  def __init__(
    self,
    f_dropout_rate,
    g_dropout_rate,
    hidden_dim,
    num_attention_heads,
    use_layer_norm,
    use_batch_norm,
    use_residual,
    use_envelope=True,
    use_multi_obj=True
  ):
    super().__init__()
    self.__name__ = "DeepRLIForInterpretation"
    self.use_multi_obj = use_multi_obj
    self.embedding_v = torch.nn.Linear(39, hidden_dim, bias=True)
    self.embedding_e = torch.nn.Linear(39, hidden_dim, bias=True)

    self.f_feat_dropout = torch.nn.Dropout(f_dropout_rate)

    self.graph_layers = torch.nn.ModuleList([GraphTransformerEdgeLayer(hidden_dim, hidden_dim, num_attention_heads, g_dropout_rate,
                        use_layer_norm, use_batch_norm, use_residual, use_envelope) for _ in range(10)])
    
    self.readout1 = MLPBlock(hidden_dim, 1)
    self.readout2 = MLPBlock(hidden_dim, 4)
    self.readout3 = MLPBlock(hidden_dim, 4)
    self.readout4 = MLPBlock(hidden_dim, 1)

  def V_steric_attraction(self, d):
    return -0.045 * torch.exp(-(d / 0.8)**2)
  
  def V_steric_repulsion(self, d):
    return 0.8 * (d < 0) * d**2
  
  def V_hydrophobic(self, d):
    return -0.035 * ((d > 0) * (d < 2.5) * (-0.4 * (d - 2.5)) + (d <= 0) * torch.ones_like(d))
  
  def V_hbond(self, d):
    return -0.6 * ((d > -0.6) * (d < 0) * (-5/3 * d) + (d <= -0.6) * torch.ones_like(d))

  def forward(self, inputs):
    g = inputs["complex.graph"]
    N_rot = inputs["ligand.num_rotatable_bonds"]
    g.apply_edges(calc_reduced_dist)
    
    v = self.embedding_v(g.ndata["feature"])
    v = self.f_feat_dropout(v)
    e = self.embedding_e(g.edata["feature"])
    for graph_layer in self.graph_layers:
      v, e = graph_layer(g, v, e)

    v = v * g.ndata["feature"][:, 0][..., None]
    g.ndata['v'] = v

    z = dgl.sum_nodes(g, 'v')
    y1 = self.readout1(z).squeeze(-1)

    g.apply_edges(u_v_add)
    V = [
      self.V_steric_attraction(g.edata["reduced_distance"]),
      self.V_steric_repulsion(g.edata["reduced_distance"]),
      self.V_hydrophobic(g.edata["reduced_distance"]),
      self.V_hbond(g.edata["reduced_distance"])
    ]

    inputs["V"] = V

    w2 = F.sigmoid(self.readout2(g.edata['x'])) + 0.5
    y2 = (g.edata["distance"] < 6.5) * g.edata["feature"][:, 0] * (
      w2[:, 0] * V[0] + w2[:, 1] * V[1] +
      g.edata["interaction_type"][:, 1] * w2[:, 2] * V[2] +
      g.edata["interaction_type"][:, 2] * w2[:, 3] * V[3]
    )
    g.edata['y2'] = y2
    y2 = dgl.sum_edges(g, 'y2') / 2

    w3 = F.sigmoid(self.readout3(g.edata['x'])) + 0.5
    y3 = (g.edata["distance"] < 6.5) * g.edata["feature"][:, 0] * (
      w3[:, 0] * V[0] + w3[:, 1] * V[1] +
      g.edata["interaction_type"][:, 1] * w3[:, 2] * V[2] +
      g.edata["interaction_type"][:, 2] * w3[:, 3] * V[3]
    )
    g.edata['y3'] = y3
    y3 = dgl.sum_edges(g, 'y3') / 2
    y3 /= 1 + (F.sigmoid(self.readout4(z)).squeeze(-1) + 0.5) * 0.05846 * N_rot

    inputs["scores"] = [y1, y2, y3]
    inputs["y3.w5"] = (F.sigmoid(self.readout4(z)).squeeze(-1) + 0.5) * 0.05846

    return inputs