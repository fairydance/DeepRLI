import torch
import torch.nn as nn
from .deeprli import DeepRLI

class ContrastiveNet(nn.Module):
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
    self.__name__ = "ContrastiveNet"
    self.use_multi_obj = use_multi_obj
    self.potential = DeepRLI(
      f_dropout_rate=f_dropout_rate,
      g_dropout_rate=g_dropout_rate,
      hidden_dim=hidden_dim,
      num_attention_heads=num_attention_heads,
      use_layer_norm=use_layer_norm,
      use_batch_norm=use_batch_norm,
      use_residual=use_residual,
      use_envelope=use_envelope,
      use_multi_obj=use_multi_obj
    )

  def forward(self, inputs):
    return [self.potential(input) for input in inputs]
