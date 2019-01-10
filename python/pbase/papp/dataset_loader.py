import logging


class DatasetLoader(object):
  """Dataset Loader:

    A unified interface for dataset loader
    To implement a dataset loader, you need to extend this class,
      and implement the `__call__` function.
    `__call__` function will be called for reading `training`, `validation`
      and `test` dataset.
  """

  def __init__(self, logger_desc):
    self.LOGGER = logging.getLogger(logger_desc + "[DatasetLoader]")
    self.LOGGER.setLevel(logging.INFO)

  def __call__(self, filenames, attributes):
    raise NotImplementedError("Please implement Dataset Loader")