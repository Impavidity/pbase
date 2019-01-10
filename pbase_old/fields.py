import six
from torchtext.data.pipeline import Pipeline
from torchtext.data.utils import get_tokenizer
import torch
from torch.autograd.variable import Variable

class Field3D(object):
    def __init__(self, init_token=None, eos_token=None,
                 tokenize1d=(lambda s: s.split(' <snippet> ')),
                 tokenize2d=(lambda s: s.split(' <eos> ')),
                 tokenize3d=(lambda s: s.split()), batch_first=False,
                 pad_token='0.0'):
        self.init_token = init_token
        self.eos_token = eos_token
        self.tokenize1d = get_tokenizer(tokenize1d)
        self.tokenize2d = get_tokenizer(tokenize2d)
        self.tokenize3d = get_tokenizer(tokenize3d)
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.use_vocab = False

    def numericalize(self, arr, device=None, train=True):
        """
        Turn a batch of examples into a Variable
        :param arr: List of tokenized and padded examples
        :param device: Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU
        :param train: Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True
        :return:
        """
        if isinstance(arr, tuple):
            arr, lengths = arr
        arr = [[float(x) for x in ex] for ex in arr]
        arr = torch.LongTensor(arr)
        if not self.batch_first:
            arr.t_()
        if device== -1:
            arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
        return Variable(arr, volatile=not train)

    def process(self, batch, device, train):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            data (torch.autograd.Varaible): Processed object given the input
                and custom postprocessing Pipeline.
        """
        #padded = self.pad(batch)
        padded = batch
        tensor = self.numericalize(padded, device=device, train=train)
        return tensor

    def preprocess(self, x):
        if (six.PY2 and isinstance(x, six.string_types) and not
                isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        # This is design for numerical tensor with different length,
        # So this string will be tokenized
        array = self.tokenize1d(x)
        maxtrix = [self.tokenize2d(a) for a in array]
        tensor = [[self.tokenize3d(t) for t in m] for m in maxtrix]
        return tensor
