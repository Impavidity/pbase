import os
import random

from pbase.papp import PYTORCH, TENSORFLOW

import numpy as np
import tensorflow as tf
import torch


class Tester(object):
  def __init__(self, config, attributes):
    self.config = config
    self.attributes = attributes

    """Set random seed"""
    if self.config.framework == TENSORFLOW:
      tf.set_random_seed(self.config.seed)
      if self.config.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
      else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
      config = tf.ConfigProto(log_device_placement=False)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
    elif self.config.framework == PYTORCH:
      torch.manual_seed(self.config.seed)
      torch.backends.cudnn.deterministic = True
      if not self.config.cuda:
        self.config.gpu = -1
      if torch.cuda.is_available() and self.config.cuda:
        torch.cuda.set_device(int(self.config.gpu))
        torch.cuda.manual_seed(self.config.seed)
    else:
      raise TypeError("{} framework is not supported".format(self.config.framework))
    np.random.seed(self.config.seed)
    random.seed(self.config.seed)

  def prepare(self, model, tensorflow_prepare=None):
    if self.config.framework == TENSORFLOW:
      self.model = model(self.config)
      self.sess.run(tf.initialize_all_variables())
      self.saver = tf.train.Saver()
      if callable(tensorflow_prepare):
        tensorflow_prepare(self.sess, self.model)
    elif self.config.framework == PYTORCH:
      self.model = model(self.config)
    else:
      raise TypeError("{} framework is not supported".format(self.config.framework))

  def restore(self):
    if self.config.framework == TENSORFLOW:
      self.saver.restore(self.sess, self.config.restore_from)
      print("Model Restored from {}".format(self.config.restore_from))
    if self.config.framework == PYTORCH:
      model_file = torch.load(self.config.restore_from)
      self.model.model_definition.load_state_dict(model_file["state_dict"])
      self.epoch = model_file['epoch']
      self.model.optimizer.load_state_dict(model_file["optimizer_state"])
      self.best_metrics = model_file["best_metrics"]
      print("Model Restored from {}".format(self.config.restore_from))
