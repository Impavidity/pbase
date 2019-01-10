import torch
from torch import nn


class CharEmbedding(nn.Embedding):
  def forward(self, input):
    return torch.stack([super(CharEmbedding, self).forward(input[i, :, :])
                        for i in range(input.size(0))], dim=0)
