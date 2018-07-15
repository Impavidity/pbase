from pbase.ptext.example import Example


class Dataset(object):
  """Define a dataset composed of Examples.

  Attributes:
    examples: List of Examples
    attributes: List of attributes
  """

  def __init__(self, examples, attributes):
    """Create a dataset from a list of dict
    """
    self.examples = [Example(ex, attributes=attributes) for ex in examples]
    self.attributes = attributes

  def __getitem__(self, item):
    return self.examples[item]

  def __len__(self):
    try:
      return len(self.examples)
    except TypeError:
      raise TypeError("function 'len' does not "
        "support type {}".format(type(self.examples)))

  def __iter__(self):
    for ex in self.examples:
      yield ex