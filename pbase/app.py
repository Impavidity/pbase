from argparse import ArgumentParser
from torchtext import data
import torch
import numpy as np
import random
import os
import sys
import uuid



class TrainAPP:
    def __init__(self, args,  fields, include_test,
                 batch_size_fn_train = lambda new, count, sofar: count,
                 batch_size_fn_valid = lambda new, count, sofar: count,
                 batch_size_fn_test = lambda new, count, sofar: count,
                 train_shuffle=True,
                 sort_within_batch=False):
        self.args = args
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        if not self.args.cuda:
            self.args.gpu = -1
        if torch.cuda.is_available() and self.args.cuda:
            print("Note: You are using GPU for training")
            torch.cuda.set_device(self.args.gpu)
            torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available() and not self.args.cuda:
            print("Warning: You have Cuda but not use it. You are using CPU for training.")
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        self.fields = fields
        train, valid, test = data.TabularDataset.splits(path=self.args.dataset_path,
                                   train=self.args.train_txt, validation=self.args.valid_txt, test=self.args.test_txt,
                                   format='TSV', fields=self.fields)
        for i in range(len(self.fields)):
            setattr(self, self.fields[i][0], self.fields[i][1])
            if self.fields[i][1].use_vocab:
                if include_test[i]:
                    self.fields[i][1].build_vocab(train, valid, test)
                else:
                    self.fields[i][1].build_vocab(train, valid)
        self.train_iter = data.Iterator(train, batch_size=self.args.batch_size, device=self.args.gpu,
                                        batch_size_fn=batch_size_fn_train, train=True, repeat=False, sort=False,
                                        shuffle=train_shuffle, sort_within_batch=sort_within_batch)
        self.valid_iter = data.Iterator(valid, batch_size=self.args.batch_size, device=self.args.gpu,
                                        batch_size_fn=batch_size_fn_valid, train=False, repeat=False, sort=False,
                                        shuffle=False, sort_within_batch=sort_within_batch)
        self.test_iter = data.Iterator(test, batch_size=self.args.batch_size, device=self.args.gpu,
                                       batch_size_fn=batch_size_fn_test, train=False, repeat=False, sort=False,
                                       shuffle=False, sort_within_batch=sort_within_batch)
        self.config = self.args


    def prepare(self, model, optimizer, criterion, evaluator,
                 metrics_comparison, log_printer):
        self.model = model(self.config)
        if self.args.cuda:
            self.model.cuda(self.args.gpu)
            print("Shift model to GPU")
        self.parameter = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optimizer(self.parameter, self.config)
        self.criterion = criterion
        self.evaluator = evaluator
        self.metrics_comparison = metrics_comparison
        os.makedirs(self.args.save_path, exist_ok=True)
        self.snapshot_path = os.path.join(self.args.save_path, self.args.prefix+"_best_model_"+str(uuid.uuid4()))
        self.patience = self.args.patience # TODO: calculate the patience
        self.log_printer = log_printer


    def train(self):
        early_stop = False
        epoch = 0
        iterations = 0
        best_metrics = None
        iters_not_improved = 0
        batch_num = len(self.train_iter)
        while True:
            if early_stop:
                print("Early Stoping")
                break
            epoch += 1
            self.train_iter.init_epoch()
            self.optimizer.schedule()
            for batch_idx, batch in enumerate(self.train_iter):
                iterations += 1
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(batch)
                # We generate metrics for each batch, not all batches so far
                metrics = self.evaluator("train", (output, batch))
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()

                if iterations % self.args.valid_every == 1:
                    self.model.eval()
                    self.valid_iter.init_epoch()
                    valid_result = []
                    for valid_batch_idx, valid_batch in enumerate(self.valid_iter):
                        valid_output = self.model(valid_batch)
                        valid_result.append((valid_output, valid_batch))
                    valid_metrics = self.evaluator("valid", valid_result)
                    self.log_printer("valid", metrics=valid_metrics, loss=loss)
                    if self.metrics_comparison(valid_metrics, best_metrics):
                        iters_not_improved = 0
                        best_metrics = valid_metrics
                        torch.save(self.model, self.snapshot_path)
                        print("Saving model to {}".format(self.snapshot_path))
                    else:
                        iters_not_improved += 1
                        if iters_not_improved >= self.patience:
                            early_stop = True
                            break

                if iterations % self.args.log_every == 0:
                    self.log_printer("train", epoch=epoch, iters=batch_idx/batch_num, metrics=metrics, loss=loss)



class TestAPP:
    def __init__(self, args, fields, include_test,
                 batch_size_fn_train=lambda new, count, sofar: count,
                 batch_size_fn_valid=lambda new, count, sofar: count,
                 batch_size_fn_test=lambda new, count, sofar: count):
        self.args = args
        if not self.args.trained_model:
            print("Error: You need to provide a option 'trained_model' to load the model")
            sys.exit(1)

        self.fields = fields
        train, valid, test = data.TabularDataset.splits(path=self.args.dataset_path,
                                                        train=self.args.train_txt, validation=self.args.valid_txt,
                                                        test=self.args.test_txt,
                                                        format='TSV', fields=self.fields)
        for i in range(len(self.fields)):
            setattr(self, self.fields[i][0], self.fields[i][1])
            if self.fields[i][1].use_vocab:
                if include_test[i]:
                    self.fields[i][1].build_vocab(train, valid, test)
                else:
                    self.fields[i][1].build_vocab(train, valid)
        self.train_iter = data.Iterator(train, batch_size=self.args.batch_size, device=self.args.gpu,
                                        batch_size_fn=batch_size_fn_train, train=True, repeat=False, sort=False,
                                        shuffle=False, sort_within_batch=False)
        self.valid_iter = data.Iterator(valid, batch_size=self.args.batch_size, device=self.args.gpu,
                                        batch_size_fn=batch_size_fn_valid, train=False, repeat=False, sort=False,
                                        shuffle=False, sort_within_batch=False)
        self.test_iter = data.Iterator(test, batch_size=self.args.batch_size, device=self.args.gpu,
                                       batch_size_fn=batch_size_fn_test, train=False, repeat=False, sort=False,
                                       shuffle=False, sort_within_batch=False)

        if self.args.cuda:
            self.model = torch.load(self.args.trained_model, map_location=lambda storage,
                                                                                 location: storage.cuda(self.args.gpu))
        else:
            self.model = torch.load(self.args.trained_model, map_location=lambda storage, location: storage)


    def prepare(self, evaluator, log_printer, output_parser=None):
        self.evaluator = evaluator
        self.log_printer = log_printer
        self.output_parser = output_parser

    def predict(self, dataset_iter, dataset_name):
        print("Dataset : {}".format(dataset_name))
        self.model.eval()
        dataset_iter.init_epoch()
        test_result = []
        for test_batch_idx, test_batch in enumerate(dataset_iter):
            results = self.model(test_batch)
            test_result.append((results, test_batch))
        test_metrics = self.evaluator(dataset_name, test_result)
        self.log_printer(dataset_name, test_metrics)
        if self.output_parser != None:
            os.makedirs(self.args.result_path, exist_ok=True)
            self.output_parser(dataset_name, test_result)



    def test(self):
        self.predict(dataset_iter=self.valid_iter, dataset_name='valid')
        self.predict(dataset_iter=self.test_iter, dataset_name='test')


class ArgParser:
    def __init__(self, description, gpu=0, batch_size=32, seed=3435,
                 dev_every=300, log_every=30, patience=5,
                 dataset_path='data', train_txt='train.txt', valid_txt='valid.txt', test_txt='test.txt',
                 save_path='saves', result_path='results'):
        self.parser = ArgumentParser(description=description)
        self.parser.add_argument('--no_cuda', action='store_false', dest='cuda')
        self.parser.add_argument('--gpu', type=int, default=gpu)
        self.parser.add_argument('--batch_size', type=int, default=batch_size)
        self.parser.add_argument('--seed', type=int, default=seed)
        self.parser.add_argument('--valid_every', type=int, default=dev_every)
        self.parser.add_argument('--log_every', type=int, default=log_every)
        self.parser.add_argument('--patience', type=int, default=patience)
        self.parser.add_argument('--dataset_path', type=str, default=dataset_path)
        self.parser.add_argument('--train_txt', type=str, default=train_txt)
        self.parser.add_argument('--valid_txt', type=str, default=valid_txt)
        self.parser.add_argument('--test_txt', type=str, default=test_txt)
        self.parser.add_argument('--save_path', type=str, default=save_path)
        self.parser.add_argument('--prefix', type=str, default="exp")
        # Tester
        self.parser.add_argument('--trained_model', type=str, default='')
        self.parser.add_argument('--result_path', type=str, default=result_path)
        self.parser.add_argument('--output_valid', type=str, default='valid.txt')
        self.parser.add_argument('--output_test', type=str, default='test.txt')



    def get_args(self):
        self.args = self.parser.parse_args()
        return self.args



