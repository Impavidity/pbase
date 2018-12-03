import math

from pbase.ptext.utils import RandomShuffler
from pbase.ptext.batch import Batch


class Iterator(object):
  """Define an iterator to generate batch data

  Attributes:
    dataset: Dataset
    batch_size: Int
    sort_key: Lambda/Function
    batch_size_fn: Lambda/Function, used for dynamic batch size
    shuffle: Boolean, whether to shuffle the dataset before each epoch
    sort: Boolean, whether to sort the dataset regarding the sort_key
      before each epoch
  """

  def __init__(self,
               dataset,
               batch_size,
               sort_key=None,
               batch_size_fn=None,
               shuffle=None,
               sort=None,
               shuffle_in_batch=False):
    self.dataset, self.batch_size = dataset, batch_size
    self.sort_key = sort_key
    self.sort = sort
    self.shuffle = shuffle
    self.random_shuffler = RandomShuffler()
    self.batch_size_fn = batch_size_fn
    self.length = None
    self.shuffle_in_batch = shuffle_in_batch

  def data(self):
    "Return the examples in the dataset in order, sorted or shuffled"
    if self.sort:
      dataset = sorted(self.dataset, key=self.sort_key)
    elif self.shuffle:
      dataset = [self.dataset[i] for i in self.random_shuffler(range(len(self.dataset)))]
    else:
      dataset = self.dataset
    return dataset

  def init_epoch(self):
    self.create_batches()

  def create_batches(self):
    self.batches = batch(self.data(), self.batch_size, self.batch_size_fn)

  def __iter__(self):
    while True:
      self.init_epoch()
      for idx, minibatch in enumerate(self.batches):
        if self.shuffle_in_batch:
          minibatch = [minibatch[i] for i in self.random_shuffler(range(len(minibatch)))]
        yield Batch(minibatch, self.dataset.attributes)
      break

  def __len__(self):
    if self.length is None:
      if self.batch_size_fn is not None:
        self.length = 0
        self.init_epoch()
        for _ in self.batches:
          self.length += 1
      else:
        self.length = math.ceil(len(self.dataset) / self.batch_size)
    return self.length


def batch(data, batch_size, batch_size_fn=None):
  """Yield elements from data in chunks of batch_size
  """
  if batch_size_fn is None:
    def batch_size_fn(new, count, sofar):
      return count
  minibatch, size_so_far = [], 0
  for ex in data:
    minibatch.append(ex)
    size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
    if size_so_far == batch_size:
      yield minibatch
      minibatch, size_so_far = [], 0
    elif size_so_far > batch_size:
      yield minibatch[:-1]
      minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
  if minibatch:
    yield minibatch