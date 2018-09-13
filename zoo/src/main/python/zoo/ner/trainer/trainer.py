from __future__ import division
import json
import os
import subprocess

from pbase.ptext.field import Field, NestedField
from pbase.ptext.attribute import Attribute
from pbase.papp.argument import Argument
from pbase.papp.trainer import Trainer
from zoo.ner.model.ner_tensorflow import NERModel


pid = os.getpid()


class NERArgument(Argument):
  def __init__(self):
    super(NERArgument, self).__init__(
      description="NER model",
      train_file="zoo/src/main/python/zoo/ner/data/en-train.json",
      valid_file="zoo/src/main/python/zoo/ner/data/en-valid.json",
      test_file="zoo/src/main/python/zoo/ner/data/en-test.json",
      framework="tensorflow",
      dev_every=3000,
      epoch=15)
    self.parser.add_argument("--word_embedding_dim", type=int, default=300)
    self.parser.add_argument("--char_embedding_dim", type=int, default=50)
    self.parser.add_argument("--padding_token", type=str, default=u"<pad>")
    self.parser.add_argument("--unk_token", type=str, default=u"<unk>")
    self.parser.add_argument("--word_vocab_path", type=str, default="data/word_vocab.pkl")
    self.parser.add_argument("--word_vocab_vector_path", type=str, default="/data/fastText/wiki.en.vec")
    self.parser.add_argument("--char_vocab_path", type=str, default="data/char_vocab.pkl")
    self.parser.add_argument("--label_vocab_path", type=str, default="data/label_vocab.pkl")
    self.parser.add_argument("--word_vocab_export_path", type=str, default="data/word_vocab.txt")
    self.parser.add_argument("--char_vocab_export_path", type=str, default="data/char_vocab.txt")
    self.parser.add_argument("--label_vocab_export_path", type=str, default="data/label_vocab.txt")
    self.parser.add_argument("--lstm_num_units", type=int, default=300)
    self.parser.add_argument("--lr", type=float, default=0.0005)
    self.parser.add_argument("--optim", type=str, default="adam")
    self.parser.add_argument("--clip", type=int, default=3)
    self.parser.add_argument("--loss_func", type=str, default="crf")


def main(args):
  print(args)
  def metrics_comparison(new_metrics, best_metrics):
    if best_metrics == None or new_metrics[2] >= best_metrics[2]:
      return True
    return False
  def evaluator(tag, results):
    tmp_file = os.path.join(args.tmp_dir, "{}.{}".format(pid, tag))
    with open(tmp_file, "w") as fout:
      for pair in results:
        batch_output = pair[0]
        batch_info = pair[1]
        for predicted, example in zip(batch_output, batch_info):
          for token, gold_label, predicted_label in zip(example.word, example.label, predicted):
            fout.write("{} {} {}\n".format(token.encode('utf-8'), gold_label.encode('utf-8'), predicted_label.encode('utf-8')))
          fout.write("\n")
    conll_eval_path = 'infrastructure/src/main/python/pbase/peval/resource/conlleval.pl'
    conll_out = subprocess.check_output(['perl {} < {}'.format(conll_eval_path, tmp_file)], shell=True)
    conll_out = conll_out.decode('utf-8').strip().split('\n')[1].split()
    f1 = float(conll_out[-1]) / 100
    p = float(conll_out[-5].strip('%;')) / 100
    r = float(conll_out[-3].strip('%;')) / 100
    return p, r, f1
  def tensorflow_prepare(sess, model):
    model.word_embedding.init(sess)
  def twitter_json_string_loader(filename, attributes):
    data = []
    with open(filename, "r") as fin:
      for line in fin:
        data.append(json.loads(line))
        for i in range(len(data[-1]["labels"])):
          data[-1]["labels"][i] = data[-1]["labels"][i].replace("O_NOT_AN_ENTITY", "O")
          data[-1]["labels"][i] = data[-1]["labels"][i].replace("_", "-")
    return data[:5000]

  WORD_FIELD = Field(sequential=True, lower=True, include_lengths=True, dump_path=args.word_vocab_path)
  CHAR_FIELD = NestedField(
    Field(sequential=True, lower=True, tokenize=list, pad_token='<c>', include_lengths=True),
    include_lengths=True, dump_path=args.char_vocab_path)
  LABEL_FIELD = Field(sequential=True, lower=False, include_lengths=True, dump_path=args.label_vocab_path)
  WORD_ATTRIBUTE = Attribute(
    target="word",
    source="tokens",
    field=WORD_FIELD,
    include_valid=True,
    include_test=True)
  CHAR_ATTRIBUTE = Attribute(
    target="char",
    source="tokens",
    field=CHAR_FIELD,
    include_valid=True,
    include_test=True)
  LABEL_ATTRIBUTE = Attribute(
    target="label",
    source="labels",
    field=LABEL_FIELD,
    include_valid=True,
    include_test=True)
  attributes = [WORD_ATTRIBUTE, CHAR_ATTRIBUTE, LABEL_ATTRIBUTE]
  trainer = Trainer(args=args, data_loader=twitter_json_string_loader, attributes=attributes)
  print("Trainer was built: {}".format(trainer))
  trainer.prepare(
    model=NERModel,
    evaluator=evaluator,
    metrics_comparison=metrics_comparison,
    tensorflow_prepare=None if args.test else tensorflow_prepare)
  if args.test:
    assert args.restore_from != ""
    trainer.restore()
    trainer.evaluate()
  else:
    trainer.train()
    inputs = {
      "char_input": trainer.model.char,
      "word_input": trainer.model.word
    }
    outputs = {
      "decode_sequence": trainer.model.decode_sequence
    }
    trainer.save(inputs=inputs, outputs=outputs)


if __name__=="__main__":
  argparser = NERArgument()
  main(argparser.get_args())