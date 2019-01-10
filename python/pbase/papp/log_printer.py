import logging

from pbase.papp import TRAIN_TAG, VALID_TAG, TEST_TAG
from pbase.papp.logger import Logger


class LogPrinter(object):
  def __init__(self, tensorboard, logger_desc):
    self.tensorboard_logger = Logger(tensorboard)
    self.LOGGER = logging.getLogger(logger_desc + "[LogPrinter]")
    self.LOGGER.setLevel(logging.INFO)

  def metrics_string(self, metrics):
    raise NotImplementedError("Please implement metrics string")

  def __call__(self, tag, metrics, loss, epoch=None, iters=None):
    if tag == TRAIN_TAG:
      self.LOGGER.info("{}\tEpoch:{}\tIter:{}\tNearest batch training Loss:{}".format(tag, epoch, iters, loss))
      metrics_string = self.metrics_string(metrics)
      if metrics_string:
        self.LOGGER.info(metrics_string)
      step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (epoch - 1)
      self.tensorboard_logger.scalar_summary(tag='loss', value=loss, step=step)
    else:
      self.LOGGER.info("{}\tLoss:{}".format(tag, loss))
      self.LOGGER.info(self.metrics_string(metrics))
      if iters != None and epoch != None and loss != None:
        step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (epoch - 1)
        self.tensorboard_logger.scalar_summary(tag='{}_loss'.format(tag), value=loss, step=step)