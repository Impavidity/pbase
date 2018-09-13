import os
import time
import random

from pbase.ptext.dataset import Dataset
from pbase.ptext.iterator import Iterator
from pbase.ptext.utils import build_vocab
from pbase.papp import PYTORCH, TENSORFLOW
from pbase.papp import TEST_TAG, TRAIN_TAG, VALID_TAG

import numpy as np
import tensorflow as tf
import torch


class Trainer(object):
  def __init__(
        self,
        args,
        data_loader,
        attributes,
        batch_size_fn_train=None,
        batch_size_fn_valid=None,
        batch_size_fn_test=None,
        train_shuffle=True,
        evaluation=False
  ):
    self.args = args
    self.attributes = attributes
    self.evaluation = evaluation

    """Set random seed"""
    if self.args.framework == TENSORFLOW:
      tf.set_random_seed(self.args.seed)
      if self.args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
      else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
      config = tf.ConfigProto(log_device_placement=False)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
    elif self.args.framework == PYTORCH:
      torch.manual_seed(self.args.seed)
      torch.backends.cudnn.deterministic = True
      if not self.args.cuda:
        self.args.gpu = -1
      if torch.cuda.is_available() and self.args.cuda:
        torch.cuda.set_device(int(self.args.gpu))
        torch.cuda.manual_seed(self.args.seed)
    else:
      raise TypeError("{} framework is not supported".format(self.args.framework))
    np.random.seed(self.args.seed)
    random.seed(self.args.seed)

    train_examples = data_loader(args.train_file, attributes)
    valid_examples = data_loader(args.valid_file, attributes)
    test_examples = data_loader(args.test_file, attributes)

    self.train_dataset = Dataset(examples=train_examples, attributes=attributes)
    self.valid_dataset = Dataset(examples=valid_examples, attributes=attributes)
    self.test_dataset = Dataset(examples=test_examples, attributes=attributes)

    build_vocab(
      attributes=self.attributes,
      train_dataset=self.train_dataset,
      valid_dataset=self.valid_dataset,
      test_dataset=self.test_dataset)

    self.train_iter = Iterator(
      dataset=self.train_dataset,
      batch_size=args.batch_size,
      batch_size_fn=batch_size_fn_train,
      shuffle=train_shuffle)
    self.valid_iter = Iterator(
      dataset=self.valid_dataset,
      batch_size=args.batch_size,
      batch_size_fn=batch_size_fn_valid,
      shuffle=False)
    self.test_iter = Iterator(
      dataset=self.test_dataset,
      batch_size=args.batch_size,
      batch_size_fn=batch_size_fn_test,
      shuffle=False)


  def prepare(self, model, evaluator, metrics_comparison, log_printer, tensorflow_prepare=None):
    if self.args.framework == TENSORFLOW:
      self.model = model(self.args)
      self.sess.run(tf.initialize_all_variables())
      self.saver = tf.train.Saver()
      if callable(tensorflow_prepare):
        tensorflow_prepare(self.sess, self.model)
    elif self.args.framework == PYTORCH:
      self.model = model(self.args)
    else:
      raise TypeError("{} framework is not supported".format(self.args.framework))
    self.evaluator = evaluator
    self.metrics_comparison = metrics_comparison
    self.log_printer = log_printer
    if not os.path.exists(self.args.tmp_dir):
      os.makedirs(self.args.tmp_dir)


  def train(self):
    epoch = 0
    iteration = 0
    best_metrics = None
    train_batch_counter = len(self.train_iter)
    while True:
      if self.args.epoch > 0 and epoch >= self.args.epoch:
        break
      self.model.schedule(epoch)
      start_training_time = time.time()
      train_loss_sum = 0
      for train_batch_idx, train_batch in enumerate(self.train_iter):
        if self.args.framework == TENSORFLOW:
          train_output, train_loss = self.model.train(train_batch, self.sess)
        elif self.args.framework == PYTORCH:
          train_output, train_loss = self.model.train(train_batch)
        else:
          raise TypeError("{} framework is not supported".format(self.args.framework))
        iteration += 1
        train_metrics = self.evaluator(TRAIN_TAG, (train_output, train_batch))
        if iteration % self.args.valid_every == 1:
          valid_results = []
          valid_loss_sum = 0
          for valid_batch_idx, valid_batch in enumerate(self.valid_iter):
            if self.args.framework == TENSORFLOW:
              valid_output, valid_loss = self.model.inference(valid_batch, self.sess)
              valid_loss_sum += valid_loss
            elif self.args.framework == PYTORCH:
              valid_output, valid_loss = self.model.inference(valid_batch)
              valid_loss_sum += valid_loss
            else:
              raise TypeError("{} framework is not supported".format(self.args.framework))
            valid_results.append((valid_output, valid_batch))
          valid_metrics = self.evaluator(VALID_TAG, valid_results)
          self.log_printer(VALID_TAG, loss=valid_loss_sum, metrics=valid_metrics)
          if self.metrics_comparison(valid_metrics, best_metrics):
            best_metrics = valid_metrics
            if self.evaluation:
              self.evaluate()
            if self.args.framework == TENSORFLOW:
              self.saver.save(self.sess, self.args.checkpoint)

        if iteration % self.args.log_every == 0:
          self.log_printer(
            TRAIN_TAG,
            loss=train_loss,
            metrics=train_metrics,
            epoch=epoch,
            iters="{}/{}".format(train_batch_idx, train_batch_counter))

      end_training_time = time.time()
      elapsed = end_training_time - start_training_time
      print("Training Epoch Time {}".format(elapsed))
      epoch += 1

  def save(self, inputs, outputs):
    if self.args.framework == TENSORFLOW:
      # Restore the best resutls
      # Simple Save
      self.saver.restore(self.sess, self.args.checkpoint)
      print("Checkpoint Restored")
      tf.saved_model.simple_save(
        self.sess, self.args.save_path, inputs, outputs
      )
      print("Model Saved")

  def restore(self):
    if self.args.framework == TENSORFLOW:
      self.saver.restore(self.sess, self.args.restore_from)
      print("Model Restored")

  def evaluate(self):
    test_results = []
    test_loss_sum = 0
    for test_batch_idx, test_batch in enumerate(self.test_iter):
      if self.args.framework == TENSORFLOW:
        test_output, test_loss = self.model.inference(test_batch, self.sess)
        test_loss_sum += test_loss
      elif self.args.framework == PYTORCH:
        test_output, test_loss = self.model.inference(test_batch)
        test_loss_sum += test_loss
      else:
        raise TypeError("{} framework is not supported".format(self.args.framework))
      test_results.append((test_output, test_batch))
    test_metrics = self.evaluator(TEST_TAG, test_results)
    self.log_printer(TEST_TAG, loss=test_loss_sum, metrics=test_metrics)