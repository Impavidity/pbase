class Pipeline(object):
  """Define a pipeline for transforming sequence data

  The input is assumed to be utf-8 encoded `str` (Python 3) or
  `unicode` (Python 2)

  Attributes:
    convert_token: Function. The function to apply to input sequence data
  """

  def __init__(self, convert_token=None):
    if convert_token is None:
      self.convert_token = self.identity
    elif callable(convert_token):
      self.convert_token = convert_token
    else:
      raise ValueError("Pipeline input convert_token {} is not None "
                       "or callable".format(convert_token))
    self.pipes = [self]

  def __call__(self, x):
    for pipe in self.pipes:
      x = pipe.call(x)
    return x

  def call(self, x):
    if isinstance(x, list):
      return [self.call(tok) for tok in x]
    return self.convert_token(x)

  def add_before(self, pipeline):
    if not isinstance(pipeline, Pipeline):
      pipeline = Pipeline(pipeline)
    self.pipes = pipeline.pipes[:] + self.pipes[:]
    return self

  def add_after(self, pipeline):
    if not isinstance(pipeline, Pipeline):
      pipeline = Pipeline(pipeline)
    self.pipes = self.pipes[:] + pipeline.pipes[:]
    return self

  @staticmethod
  def identity(x):
    return x