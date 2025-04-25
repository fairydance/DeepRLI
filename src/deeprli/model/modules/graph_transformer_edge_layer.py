import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
  Graph Transformer Layer with edge features
  
"""

"""
  Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
  def func(edges):
    return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
  return func

def scaling(field, scale_constant):
  def func(edges):
    return {field: ((edges.data[field]) / scale_constant)}
  return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
  """
    implicit_attn: the output of K Q
    explicit_edge: the explicit edge features
  """
  def func(edges):
    return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
  return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
  def func(edges):
    return {'e_out': edges.data[edge_feat]}
  return func


def exp(field):
  def func(edges):
    # clamp for softmax numerical stability
    return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
  return func


"""
  Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
  def __init__(self, in_dim, out_dim, num_heads, use_bias, use_envelope):
    super().__init__()
    
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.use_envelope = use_envelope
    
    if use_bias:
      self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
      self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
      self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
      self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
    else:
      self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
      self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
      self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
      self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
  
  def propagate_attention(self, g):
    # Compute attention score
    g.apply_edges(src_dot_dst("K_v", "Q_v", "score"))
    
    # scaling
    g.apply_edges(scaling("score", np.sqrt(self.out_dim)))
    
    # Use available edge features to modify the scores
    g.apply_edges(imp_exp_attn("score", "proj_e"))
    
    # Copy edge features as e_out to be passed to FFN_e
    g.apply_edges(out_edge_features("score"))

    # softmax
    g.apply_edges(exp("score"))

    # Send weighted values to target nodes
    eids = g.edges()
    if self.use_envelope:
      g.apply_edges(fn.u_mul_e("V_v", "envelope", "V_v"))
    g.send_and_recv(eids, fn.u_mul_e("V_v", "score", "V_v"), fn.sum("V_v", "wV"))
    g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
  
  def forward(self, g, v, e):
    
    Q_v = self.Q(v)
    K_v = self.K(v)
    V_v = self.V(v)
    proj_e = self.proj_e(e)
    
    # Reshaping into [num_nodes, num_heads, feat_dim] to 
    # get projections for multi-head attention
    g.ndata["Q_v"] = Q_v.view(-1, self.num_heads, self.out_dim)
    g.ndata["K_v"] = K_v.view(-1, self.num_heads, self.out_dim)
    g.ndata["V_v"] = V_v.view(-1, self.num_heads, self.out_dim)
    g.edata["proj_e"] = proj_e.view(-1, self.num_heads, self.out_dim)
    
    self.propagate_attention(g)
    
    v_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
    e_out = g.edata["e_out"]
    
    return v_out, e_out
  

class GraphTransformerEdgeLayer(nn.Module):
  """
    Param: 
  """
  def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False,
               batch_norm=True, residual=True, use_bias=False, use_envelope=True):
    super().__init__()

    self.in_channels = in_dim
    self.out_channels = out_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.residual = residual
    self.layer_norm = layer_norm     
    self.batch_norm = batch_norm
    
    self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias, use_envelope)
    
    self.O_v = nn.Linear(out_dim, out_dim)
    self.O_e = nn.Linear(out_dim, out_dim)

    if self.layer_norm:
      self.layer_norm1_v = nn.LayerNorm(out_dim)
      self.layer_norm1_e = nn.LayerNorm(out_dim)
      
    if self.batch_norm:
      self.batch_norm1_v = nn.BatchNorm1d(out_dim)
      self.batch_norm1_e = nn.BatchNorm1d(out_dim)
    
    # FFN for v
    self.FFN_v_layer1 = nn.Linear(out_dim, out_dim*2)
    self.FFN_v_layer2 = nn.Linear(out_dim*2, out_dim)
    
    # FFN for e
    self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
    self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

    if self.layer_norm:
      self.layer_norm2_v = nn.LayerNorm(out_dim)
      self.layer_norm2_e = nn.LayerNorm(out_dim)
      
    if self.batch_norm:
      self.batch_norm2_v = nn.BatchNorm1d(out_dim)
      self.batch_norm2_e = nn.BatchNorm1d(out_dim)
    
  def forward(self, g, v, e):
    v_in1 = v # for first residual connection
    e_in1 = e # for first residual connection
    
    # multi-head attention out
    v_attn_out, e_attn_out = self.attention(g, v, e)
    
    v = v_attn_out.view(-1, self.out_channels)
    e = e_attn_out.view(-1, self.out_channels)
    
    v = F.dropout(v, self.dropout, training=self.training)
    e = F.dropout(e, self.dropout, training=self.training)

    v = self.O_v(v)
    e = self.O_e(e)

    if self.residual:
      v = v_in1 + v # residual connection
      e = e_in1 + e # residual connection

    if self.layer_norm:
      v = self.layer_norm1_v(v)
      e = self.layer_norm1_e(e)

    if self.batch_norm:
      v = self.batch_norm1_v(v)
      e = self.batch_norm1_e(e)

    v_in2 = v # for second residual connection
    e_in2 = e # for second residual connection

    # FFN for v
    v = self.FFN_v_layer1(v)
    v = F.relu(v)
    v = F.dropout(v, self.dropout, training=self.training)
    v = self.FFN_v_layer2(v)

    # FFN for e
    e = self.FFN_e_layer1(e)
    e = F.relu(e)
    e = F.dropout(e, self.dropout, training=self.training)
    e = self.FFN_e_layer2(e)

    if self.residual:
      v = v_in2 + v # residual connection       
      e = e_in2 + e # residual connection  

    if self.layer_norm:
      v = self.layer_norm2_v(v)
      e = self.layer_norm2_e(e)

    if self.batch_norm:
      v = self.batch_norm2_v(v)
      e = self.batch_norm2_e(e)             

    return v, e
    
  def __repr__(self):
    return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                       self.in_channels,
                       self.out_channels, self.num_heads, self.residual)