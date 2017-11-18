from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab
from collections import Counter, OrderedDict
from torch.autograd import Variable

class CharField(Field):

    vocab_cls = Vocab

    def __init__(self, **kwargs):
        super(CharField, self).__init__(**kwargs)
        if self.preprocessing is None:
            self.preprocessing = lambda x: [list(y) for y in x]

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                for y in x:
                    counter.update(y)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def padChar(self, minibatch, max_word_len):
        minibatch = list(minibatch)
        padded = []
        for x in minibatch:
            padded.append(
                list(x[:max_word_len]) +
                [self.pad_token] * max(0, max_word_len - len(x))
            )
        return padded


    def pad(self, minibatch):
        # Not support non-sequential, fix_length, pad_first, init_token, eos_token, lengths
        minibatch = list(minibatch)
        max_sent_len = max(len(x) for x in minibatch)
        max_word_len = max(len(y) for x in minibatch for y in x)
        padded = []
        for x in minibatch:
            padded.append(
                self.padChar(x[:max_sent_len], max_word_len) +
                [[self.pad_token] * max_word_len] * max(0, max_sent_len - len(x))
            )
        return padded

    def numericalize(self, arr, device=None, train=True):
        # Definitely use vocab, so self.use_vocab = True
        # Definitely use sequential, so self.sequential = True
        arr = [[[self.vocab.stoi[y] for y in x] for x in ex] for ex in arr]
        if self.postprocessing is not None:
            arr = self.postprocessing(arr, self.vocab, train)
        arr = self.tensor_type(arr)
        if self.sequential and not self.batch_first:
            arr = arr.transpose(0,1)
        if device == -1:
            arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
        return Variable(arr, volatile=not train)



