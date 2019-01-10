import torch
from collections import defaultdict
from torchtext.data import Batch
from torchtext.data.iterator import RandomShuffler
import numpy as np

class PairIterator(object):
    """
    Defines an iterator that loads positive-negative instance pair of data from a Dataset
    This iterator is specifically for training
    """
    def __init__(self, dataset, device=None, shuffle=True, group_id=None, label_id=None,
                 positive_labels=None, negative_labels=None, pair_fn=None, sample_size=None):
        self.dataset = dataset
        self.device = device

        if not torch.cuda.is_available() and self.device is None:
            self.device = -1
        if group_id is None:
            raise ValueError("group_id is not given")
        if label_id is None:
            raise ValueError("label_id is not given")
        self.group_id = group_id
        self.label_id = label_id
        if type(positive_labels) != list:
            raise ValueError("positive_labels need to be a list")
        if type(negative_labels) != list:
            raise ValueError("negative_labels need to be a list")
        self.positive_labels = positive_labels
        self.negative_labels = negative_labels
        self.shuffle = shuffle
        self.random_shuffler = RandomShuffler()
        self.pair_fn = pair_fn
        self.sample_size = sample_size
        self.pairs = []
        self.partition()
        self.create_pairs()
        self.pair_num = len(self.pairs)

    def partition(self):
        self.positive, self.negative = defaultdict(list), defaultdict(list)
        for ex in self.dataset:
            group = getattr(ex, self.group_id)
            label = getattr(ex, self.label_id)
            if label in self.positive_labels:
                self.positive[group].append(ex)
            elif label in self.negative_labels:
                self.negative[group].append(ex)
            else:
                raise ValueError("The label of {} is not in the given list".format(label))
        if self.pair_fn is None:
            self.pair_fn = product

    def create_pairs(self):
        for group_id in self.positive.keys():
            self.pairs.extend(self.pair_fn(self.positive[group_id], self.negative[group_id]))
        if self.sample_size is not None:
            self.pairs = np.random.choice(self.pairs, self.sample_size)


    def init_epoch(self):
        if self.shuffle:
            self.pairs = [self.pairs[i] for i in self.random_shuffler(range(self.pair_num))]


    def __len__(self):
        return self.pair_num

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, pair in enumerate(self.pairs):
                yield Batch(pair.positive, self.dataset, self.device, True), \
                      Batch(pair.negative, self.dataset, self.device, True)
            break

class Pair:
    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative

def product(positive_list, negative_list):
    pairs = []
    for positive_item in positive_list:
        for negative_item in negative_list:
            pair = Pair([positive_item], [negative_item])
            pairs.append(pair)
    return pairs


