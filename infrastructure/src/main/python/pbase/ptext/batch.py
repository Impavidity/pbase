class Batch(object):
  """Define a batch of examples along with its Fields

  Attributes:
    batch_size: Number of examples in the batch
    data: List of Examples
    
  Args:
    data: 
    attributes: 
  """

  def __init__(self, data=None, attributes=None):

    if data is not None:
      self.batch_size = len(data)
      self.data = data

      for attribute in attributes:
        batch = [getattr(ex, attribute.target) for ex in data]
        setattr(self, attribute.target, attribute.field.pad(batch))

  def __len__(self):
    return self.batch_size

  def __iter__(self):
    for x in self.data:
      yield x

  def __getitem__(self, i):
    return self.data[i]
