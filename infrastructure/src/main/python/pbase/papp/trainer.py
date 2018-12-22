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
        attributes_train=None,
        attributes_valid=None,
        attributes_test=None,
        batch_size_fn_train=None,
        batch_size_fn_valid=None,
        batch_size_fn_test=None,
        train_shuffle=True,
        train_shuffle_in_batch=False,
        evaluation=False
  ):
    self.args = args
    self.attributes = attributes
    self.attributes_train = attributes_train if attributes_train else attributes
    self.attributes_valid = attributes_valid if attributes_valid else attributes
    self.attributes_test = attributes_test if attributes_test else attributes
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

    train_examples = data_loader(args.train_file, self.attributes_train)
    valid_examples = data_loader(args.valid_file, self.attributes_valid)
    test_examples = data_loader(args.test_file, self.attributes_test)

    self.train_dataset = Dataset(examples=train_examples, attributes=self.attributes_train)
    self.valid_dataset = Dataset(examples=valid_examples, attributes=self.attributes_valid)
    self.test_dataset = Dataset(examples=test_examples, attributes=self.attributes_test)

    if not args.test and not args.reuse_vocab:
      build_vocab(
        attributes=self.attributes,
        train_dataset=self.train_dataset,
        valid_dataset=self.valid_dataset,
        test_dataset=self.test_dataset)

    self.train_iter = Iterator(
      dataset=self.train_dataset,
      batch_size=args.batch_size,
      batch_size_fn=batch_size_fn_train,
      shuffle=train_shuffle,
      shuffle_in_batch=train_shuffle_in_batch)
    self.valid_iter = Iterator(
      dataset=self.valid_dataset,
      batch_size=args.batch_size,
      batch_size_fn=batch_size_fn_valid,
      shuffle=False,
      shuffle_in_batch=False)
    self.test_iter = Iterator(
      dataset=self.test_dataset,
      batch_size=args.batch_size,
      batch_size_fn=batch_size_fn_test,
      shuffle=False,
      shuffle_in_batch=False)

    self.epoch = 0
    self.iteration = 0
    self.best_metrics = None


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
    train_batch_counter = len(self.train_iter)
    while True:
      if self.args.epoch > 0 and self.epoch >= self.args.epoch:
        break
      self.model.schedule(self.epoch)
      start_training_time = time.time()
      train_loss_sum = 0
      for train_batch_idx, train_batch in enumerate(self.train_iter):
        if self.args.framework == TENSORFLOW:
          train_output, train_loss = self.model.train(train_batch, self.sess)
        elif self.args.framework == PYTORCH:
          train_output, train_loss = self.model.train(train_batch)
        else:
          raise TypeError("{} framework is not supported".format(self.args.framework))
        self.iteration += 1
        train_metrics = self.evaluator(TRAIN_TAG, (train_output, train_batch))
        if self.iteration % self.args.valid_every == 1:
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
          if self.metrics_comparison(valid_metrics, self.best_metrics):
            self.best_metrics = valid_metrics
            if self.evaluation:
              self.evaluate()
            if self.args.framework == TENSORFLOW:
              self.saver.save(self.sess, self.args.checkpoint)
            if self.args.framework == PYTORCH:
              self.save_pytorch_checkpoint(
                epoch=self.epoch,
                state_dict=self.model.model_definition.state_dict(),
                optimizer_state=self.model.optimizer.state_dict(),
                best_metrics=self.best_metrics,
                filename=self.args.checkpoint)

        if self.iteration % self.args.log_every == 0:
          self.log_printer(
            TRAIN_TAG,
            loss=train_loss,
            metrics=train_metrics,
            epoch=self.epoch,
            iters="{}/{}".format(train_batch_idx, train_batch_counter))

      end_training_time = time.time()
      elapsed = end_training_time - start_training_time
      print("Training Epoch Time {}".format(elapsed))
      self.epoch += 1

  def save(self, inputs, outputs):
    if self.args.framework == TENSORFLOW:
      # Restore the best resutls
      # Simple Save
      self.saver.restore(self.sess, self.args.checkpoint)
      print("Checkpoint Restored")
      tf.saved_model.simple_save(
        self.sess, self.args.save_path, inputs, outputs)
      print("Model Saved to {}".format(self.args.checkpoint))


  def save_pytorch_checkpoint(self, epoch, state_dict, optimizer_state, best_metrics, filename):
    for k, tensor in state_dict.items():
      state_dict[k] = tensor.cpu()
    state = {
      'epoch': epoch,
      'state_dict': state_dict,
      'optimizer_state': optimizer_state,
      'best_metrics': best_metrics}
    base_dir = os.path.dirname(filename)
    if base_dir:
      if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torch.save(state, filename)
    print("Model Saved to {}".format(filename))

  def restore(self):
    if self.args.framework == TENSORFLOW:
      self.saver.restore(self.sess, self.args.restore_from)
      print("Model Restored from {}".format(self.args.restore_from))
    if self.args.framework == PYTORCH:
      model_file = torch.load(self.args.restore_from)
      self.model.model_definition.load_state_dict(model_file["state_dict"])
      self.epoch = model_file['epoch']
      self.model.optimizer.load_state_dict(model_file["optimizer_state"])
      self.best_metrics = model_file["best_metrics"]
      print("Model Restored from {}".format(self.args.restore_from))

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