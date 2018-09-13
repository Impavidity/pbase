from argparse import ArgumentParser


class Argument(object):
  def __init__(self, description, gpu="0", batch_size=8, seed=3435,
               dev_every=300, log_every=30, patience=5, epoch=0,
               framework="pytorch",
               dataset_path='data', train_file='train.txt', valid_file='valid.txt', test_file='test.txt',
               save_path='saves', checkpoint="checkpoint/model.ckpt", result_path='results'):
    self.parser = ArgumentParser(description=description)
    self.parser.add_argument('--framework', type=str, default=framework)
    self.parser.add_argument('--no_cuda', action='store_false', dest='cuda')
    self.parser.add_argument('--gpu', type=int, default=gpu)
    self.parser.add_argument('--batch_size', type=int, default=batch_size)
    self.parser.add_argument('--seed', type=int, default=seed)
    self.parser.add_argument('--valid_every', type=int, default=dev_every)
    self.parser.add_argument('--log_every', type=int, default=log_every)
    self.parser.add_argument('--patience', type=int, default=patience)
    self.parser.add_argument('--dataset_path', type=str, default=dataset_path)
    self.parser.add_argument('--train_file', type=str, nargs="+", default=train_file)
    self.parser.add_argument('--valid_file', type=str, nargs="+", default=valid_file)
    self.parser.add_argument('--test_file', type=str, nargs="+", default=test_file)
    self.parser.add_argument('--save_path', type=str, default=save_path)
    self.parser.add_argument('--checkpoint', type=str, default=checkpoint)
    self.parser.add_argument('--prefix', type=str, default="exp")
    self.parser.add_argument("--tmp_dir", type=str, default="tmp")
    self.parser.add_argument("--epoch", type=int, default=epoch)
    # Tester
    self.parser.add_argument("--test", action="store_true", dest='test')
    self.parser.add_argument('--restore_from', type=str, default='')
    self.parser.add_argument('--result_path', type=str, default=result_path)
    self.parser.add_argument('--output_valid', type=str, default='valid.txt')
    self.parser.add_argument('--output_test', type=str, default='test.txt')

  def get_args(self, args=None):
    if args is not None:
      self.args = self.parser.parse_args(args=args)
    else:
      self.args = self.parser.parse_args()
    return self.args



