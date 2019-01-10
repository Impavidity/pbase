import json


def json_loader(filename, attributes):
  pass


def json_string_loader(filename, attributes):
  data = []
  with open(filename, "r") as fin:
    for line in fin:
      data.append(json.loads(line))
  return data


def tsv_loader(filename, attributes):
  pass


def conll_loader(filename, attributes):
  pass