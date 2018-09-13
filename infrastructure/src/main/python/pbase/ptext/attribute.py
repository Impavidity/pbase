class Attribute(object):
  """Define the attribute wrapping the
    - target: String, the attribute name in the batch
    - source: String, the key in the example dict
    - field: Field, the field for target attribute
    - include_valid: Boolean, if True, the validation
     dataset vocab will be included.
    - include_test: Boolean, if True, the testing dataset
     vocab will be included
  """

  def __init__(
        self,
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
