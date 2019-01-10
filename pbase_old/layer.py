import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Embedding(nn.Embedding):
    def reset_parameters(self):
        print("Use uniform to initialize the embedding")
        self.weight.data.uniform_(-0.05, 0.05)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class CharEmbedding(nn.Embedding):
    def forward(self, input):
        return torch.stack([super(CharEmbedding, self).forward(input[i, :, :])
                            for i in range(input.size(0))], dim=0)

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

    def forward(self, x):
        size = x.size()
        if len(size) > 2:
            y = super(MLP, self).forward(
                x.contiguous().view(-1, size[-1]))
            y = y.view(size[0:-1] + (-1,))
        else:
            y = super(MLP, self).forward(x)
        if self._dropout_ratio > 0:
            return self._dropout(self._activate(y))
        else:
            return self._activate(y)

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
                nn.init.constant(self.__getattr__(name), 0)

    def forward(self, input, hx=None):
        # TODO: actually RNN has handled with situation when hx==None
        # TODO: it will generate the zero tensor for h0 and c0
        # TODO: If you want to have a different initialization here
        return super(LSTM, self).forward(input, hx)

class SimpleCNN(nn.Module):
    def __init__(self, num_of_conv, in_channels, out_channels, kernel_size, in_features, out_features, stride=1,
                 dilation=1, groups=1, bias=True, active_func = F.relu, pooling=F.max_pool1d,
                 dropout=0.5, padding_strategy="default", padding_list=None, fc_layer=True):
        """

        :param num_of_conv: Follow kim cnn idea
        :param kernel_size: if is int type, then make it into list, length equals to num_of_conv
                     if list type, then check the length of it, should has length of num_of_conv
        :param out_features: feature size
        """
        super(SimpleCNN, self).__init__()
        if type(kernel_size) == int:
            kernel_size = [kernel_size]
        if len(kernel_size) != num_of_conv:
            print("Number of kernel_size should be same num_of_conv")
            exit(1)
        if padding_list == None:
            if padding_strategy == "default":
                padding_list = [(k_size-1, 0) for k_size in kernel_size]

        self.conv = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(k_size, in_features),
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation,
                                             groups=groups,
                                             bias=bias)
                                   for k_size, padding in zip(kernel_size, padding_list)])
        self.pooling = pooling
        self.active_func = active_func
        self.fc_layer = fc_layer
        if fc_layer:
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_of_conv * out_channels, out_features)


    def forward(self, input):
        if len(input.size()) == 3:
            input = input.unsqueeze(1)
        # input = (batch, in_channels, sent_len, word_dim)
        x = [self.active_func(conv(input)).squeeze(3) for conv in self.conv]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [self.pooling(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        x = torch.cat(x, 1) # (batch, out_channels * Ks)
        if self.fc_layer:
            x = self.dropout(x)
            x = self.fc(x)
        return x

class SequenceCNN(nn.Module):
    def __init__(self, num_of_conv, in_channels, out_channels, kernel_size, in_features, stride=1,
                 dilation=1, groups=1, bias=True):
        super(SequenceCNN, self).__init__()
        if type(kernel_size) == int:
            kernel_size = [kernel_size]
        if len(kernel_size) != num_of_conv:
            print("Number of kernel_size should be same num_of_conv")
            exit(1)
        for k_size in kernel_size:
            if k_size % 2 == 0:
                print("The kernel size is better to be odd")
                exit(1)
        padding_list = [(int(k_size / 2), 0) for k_size in kernel_size]
        self.conv = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(k_size, in_features),
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation,
                                             groups=groups,
                                             bias=bias)
                                   for k_size, padding in zip(kernel_size, padding_list)])

    def forward(self, input):
        if len(input.size()) == 3:
            input = input.unsqueeze(1)
        # input = (batch, in_channels, sent_len, word_dim)
        x = [conv(input).squeeze(3).transpose(1,2) for conv in self.conv]
        # x = (batch, sent_len, out_channels) * Ks
        x = torch.cat(x, dim=2)
        return x




class CharCNN(SimpleCNN):
    """
    Single CNN for char
    input: Tensor (batch, sent_len, word_len, char_dim)
    """
    def forward(self, input):
        if len(input.size()) == 4:
            input = input.unsqueeze(2)
        # input = (batch, sent_len, in_channels, word_len, char_dim)
        x = torch.stack([super(CharCNN, self).forward(input[i,:,:,:,:])
                         for i in range(input.size(0))], dim=0)
        # x = (batch, sent_len, output_feature)
        return x




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


class Highway(nn.Module):
    def __init__(self, hidden_size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.f = f

    def forward(self, input):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](input))
            nonlinear = self.f(self.nonlinear[layer](input))
            linear = self.linear[layer](input)
            x = gate * nonlinear + (1 - gate) * linear
        return x

class HighwayLSTM(nn.Module):
    def __init__(self):
        super(HighwayLSTM, self).__init__()
        pass


class CRF(nn.Module):
    def __init__(self, tagset_size, gpu=-1):
        super(CRF, self).__init__()
        self.average_batch = False
        self.gpu = gpu
        self.tagset_size =  tagset_size
        # We add 2 here, because of STAET_TAG and STOP_TAG
        # transition (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        if self.gpu != -1:
            init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2).cuda(self.gpu)
        else:
            init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        self.transitions = nn.Parameter(init_transitions)


    def viterbi_decode(self, in_features, mask):
        """

        :param in_features: (batch_size, sequence_length, self.tag_size + 2)
        :param mask: (batch_size, sequence_length)
        :return: decode_idx: (batch_size, sequence_length)
                 path_score: (batch_size, 1)
        """
        batch_size = in_features.size(0)
        sequence_length = in_features.size(1)
        tag_size = in_features.size(2)
        assert(tag_size == self.tagset_size + 2)
        # calculate sentence length for each sentence
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        # Transpose mask to (sequence_length, batch_size) TODO: Why
        mask = mask.transpose(1,0).contiguous()
        ins_num = sequence_length * batch_size
        feats = in_features.tranpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # TODO: Why do we need add here
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(sequence_length, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.next()







