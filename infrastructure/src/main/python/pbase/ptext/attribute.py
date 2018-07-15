class Attribute(object):
  """Define the attribute wrapping the
    - target: String, the attribute name in the batch
    - source: String, the key in the example dict
    - field: Field, the field for target attribute
  """

  def __init__(self, target, source, field):
    self.target = target
    self.source = source
    self.field = field