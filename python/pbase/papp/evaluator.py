import logging


class Evaluator(object):
  """
  The variables in eval_init have states.
  For each __call__ function, please make sure
    you clear all variables.
  """
  def __init__(self, logger_desc):
    self.LOGGER = logging.getLogger(logger_desc + "[Evaluator]")
    self.LOGGER.setLevel(logging.INFO)

  def eval_init(self, tag):
    raise NotImplementedError("Please implement eval_init")

  def eval_on_batch(self, output, batch):
    """
    Make sure that you are familiar with the type of `output` and `batch`.
    """
    raise NotImplementedError("Please implement eval_on_batch")

  def finalize(self, tag):
    raise NotImplementedError("Please implement finalize")

  def __call__(self, tag, results):
    if type(results) != list and type(results) == tuple:
      results = [results]
    if type(results) != list:
      self.LOGGER.error(
        "Output of Model should be Tuple or List[Tuple], but got {}".format(
          type(results)))
    self.eval_init(tag)
    for output, batch in results:
      self.eval_on_batch(output, batch)
    return self.finalize(tag)