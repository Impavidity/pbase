import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math


class MLP(nn.Linear):
    def __init__(self, in_features, out_features, activation=None, dropout=0.0, bias=True):
        super(MLP, self).__init__(in_features, out_features, bias)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable, but got {}".format(type(activation)))
            self._activate = activation
        assert dropout == 0 or type(dropout) == float
        self._dropout_ratio = dropout
        if dropout > 0:
            self._dropout = nn.Dropout(p=self._dropout_ratio)
        else:
            self._dropout = lambda x: x

    def forward(self, x):
        size = x.size()
        if len(size) > 2:
            y = super(MLP, self).forward(
                x.contiguous().view(-1, size[-1]))
            y = y.view(size[0:-1] + (-1,))
        else:
            y = super(MLP, self).forward(x)
        return self._dropout(self._activate(y))

class MLPs(nn.ModuleList):
    def __init__(self, layers):
        assert all(type(layer) == MLP for layer in layers)
        super(MLPs, self).__init__(layers)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        for name, param in self.named_parameters():
            if "weight" in name:
                for i in range(4):
                    nn.init.orthogonal(self.__getattr__(name)[self.hidden_size*i:self.hidden_size*(i+1),:])
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), -1)

    def forward(self, input, hx=None):
        # TODO: actually RNN has handled with situation when hx==None
        # TODO: it will generate the zero tensor for h0 and c0
        # TODO: If you want to have a different initialization here
        return super(LSTM, self).forward(input, hx)

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=(True, True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self._use_bias = bias

        shape = (in1_features + int(bias[0]),
                 in2_features + int(bias[1]),
                 out_features)
        self.weight = nn.Parameter(torch.Tensor(*shape))
        if bias[2]:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            # TODO: why not self.bias = None ?
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # According to https://github.com/tdozat/Parser-v1/blob/Definitely-working/lib/linalg.py# L97
        # The parameters are init with orthogonal
        nn.init.orthogonal(self.weight)
        if self.bias is not None:
            # TODO: need to verify the initialization method
            # stdv = 1. / math.sqrt(self.bias.size(0))
            # self.bias.data.uniform_(-stdv, stdv)
            nn.init.constant(self.bias, 0)

    def forward(self, input1, input2):
        # TODO: Might set dropout for the inputs, refer to
        # https://github.com/tdozat/Parser-v1/blob/Definitely-working/lib/models/nn.py#L303
        is_cuda = next(self.parameters()).is_cuda
        device_id = next(self.parameters()).get_device() if is_cuda else None
        out_size = self.out_features
        batch_size, len1, dim1 = input1.size()
        if self._use_bias[0]:
            ones = torch.ones(batch_size, len1, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input1 = torch.cat([input1, Variable(ones)], dim=2)
            dim1 += 1
        len2, dim2 = input2.size()[1:]
        if self._use_bias[1]:
            ones = torch.ones(batch_size, len2, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input2 = torch.cat([input2, Variable(ones)], dim=2)
            dim2 += 1
        # input1_reshape = (batch* len, dim1)
        input1_reshaped = input1.contiguous().view(batch_size * len1, dim1)
        # W_reshaped = (dim1, output_size * dim2)
        W_reshaped = torch.transpose(self.weight, 1, 2).contiguous().view(dim1, out_size * dim2)
        # (bn, d) (d, rd) -> (bn, rd) -> (b, nr, d)
        affine = torch.mm(input1_reshaped, W_reshaped).view(batch_size, len1* out_size, dim2)
        # (b, nr, d) (b, n, d)T -> (b, nr, n) -> (b, n, r, n)
        input2_transpose = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2_transpose).view(batch_size, len1, out_size, len2)
        # (b, n, r, n) -> (b, n, n, r)
        biaffine = torch.transpose(biaffine, 2, 3).contiguous()
        if self._use_bias[2]:
            biaffine += self.bias.expand_as(biaffine)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + 'in1_features=' + str(self.in1_features) \
                + ', in2_features=' + str(self.in2_features) \
                + ', out_features=' + str(self.out_features) + ')'





