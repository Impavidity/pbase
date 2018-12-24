class Attribute(object):
  """
  
  Args:
    target (str): the attribute name in the batch
    source (str): the key in the example dict
    field (:class:`Field`): the field for target attribute
    include_valid (bool): if True, the validation
      dataset vocab will be included.
    include_test (bool): if True, the testing dataset
      vocab will be included
  """

  def __init__(self,
               target,
               source,
               field,
               include_valid=False,
               include_test=False):
    self.target = target
    self.source = source
    self.field = field
    self.include_valid = include_valid
    self.include_test = include_test
