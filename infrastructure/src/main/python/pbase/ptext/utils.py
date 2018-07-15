import random


class RandomShuffler(object):

  def __call__(self, data):
    return random.sample(data, len(data))