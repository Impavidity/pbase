import json


class Config(object):
  """Load argument from json file, creating :class:`~Config` object
  """

  def __init__(self, path=None):
    if path:
      self.load(path)

  def load(self, path):
    with open(path) as fin:
      args = json.load(fin)
    for key in args:
      self.__setattr__(key, args[key])

  def test_mode(self, restore_from):
    self.restore_from = restore_from
    self.test = True