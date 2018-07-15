# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from pbase.ptext.field import Field, NestedField

import pytest


batch_1 = ["John loves Mary", "Mary cries"]
batch_padded_1 = [["John", "loves", "Mary"],
                  ["Mary", "cries", "<pad>"]]
pad_token_1 = "<pad>"
sequence_lengths_1 = [3, 2]

batch_2 = ["John loves Mary", "Mary cries"]
batch_padded_2 = [["John", "loves", "Mary"],
                  ["Mary", "cries"]]

CHAR_FIELD = Field(sequential=True,
                   init_token=None,
                   eos_token=None,
                   preprocessing=None,
                   postprocessing=None,
                   lower=True,
                   tokenize=list,
                   include_lengths=True,
                   pad_token="<c>")
batch_3 = ["John loves Mary", "Mary cries"]
batch_padded_3 = [[['j', 'o', 'h', 'n', '<c>'],
                   ['l', 'o', 'v', 'e', 's'],
                   ['m', 'a', 'r', 'y', '<c>']],
                  [['m', 'a', 'r', 'y', '<c>'],
                   ['c', 'r', 'i', 'e', 's'],
                   ['<c>', '<c>', '<c>', '<c>', '<c>']]]
word_lengths = [3, 2]
char_lengths = [[4, 5, 4], [4, 5, 0]]
sequence_lengths_3 = char_lengths
batch_4 = ["John love Mary . <split> Mary cries .",
           "Will you merry me ? <split> Yes <split> Thank !"]
WORD_FIELD = NestedField(CHAR_FIELD,
                         include_lengths=True)
SENT_FIELD = NestedField(WORD_FIELD,
                         tokenize=lambda x: x.split('<split>'),
                         include_lengths=True)
batch_padded_4 = [[[['j', 'o', 'h', 'n', '<c>'],
                    ['l', 'o', 'v', 'e', '<c>'],
                    ['m', 'a', 'r', 'y', '<c>'],
                    ['.', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>']],
                   [['m', 'a', 'r', 'y', '<c>'],
                    ['c', 'r', 'i', 'e', 's'],
                    ['.', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>']],
                   [['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>']]],
                  [[['w', 'i', 'l', 'l', '<c>'],
                    ['y', 'o', 'u', '<c>', '<c>'],
                    ['m', 'e', 'r', 'r', 'y'],
                    ['m', 'e', '<c>', '<c>', '<c>'],
                    ['?', '<c>', '<c>', '<c>', '<c>']],
                   [['y', 'e', 's', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>']],
                   [['t', 'h', 'a', 'n', 'k'],
                    ['!', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>']]]]
sequence_lengths_4 = [[[4, 4, 4, 1, 0],
                       [4, 5, 1, 0, 0],
                       [0, 0, 0, 0, 0]],
                      [[4, 3, 5, 2, 1],
                       [3, 0, 0, 0, 0],
                       [5, 1, 0, 0, 0]]]


@pytest.mark.parametrize("batch, batch_padded, sequence_lengths,"
                         "sequential, init_token, eos_token, "
                         "preprocessing, postprocessing, lower, "
                         "tokenize, include_lengths, pad_token",
                         [(batch_1, batch_padded_1, sequence_lengths_1,
                           True, None, None, None, None,
                           False, (lambda s: s.split()), True, pad_token_1)])
def test_field(batch,
               batch_padded,
               sequence_lengths,
               sequential,
               init_token,
               eos_token,
               preprocessing,
               postprocessing,
               lower,
               tokenize,
               include_lengths,
               pad_token):
  text_field = Field(sequential=sequential,
                     init_token=init_token,
                     eos_token=eos_token,
                     preprocessing=preprocessing,
                     postprocessing=postprocessing,
                     lower=lower,
                     tokenize=tokenize,
                     include_lengths=include_lengths,
                     pad_token=pad_token)
  if include_lengths:
    batch, lengths = text_field.process(batch)
    assert batch == batch_padded
    assert lengths == sequence_lengths
  else:
    batch = text_field.process(batch)
    assert batch == batch_padded


@pytest.mark.parametrize("batch, batch_padded, sequence_lengths, nesting_field, "
                         "init_token, eos_token, preprocessing, postprocessing, "
                         "tokenize, include_lengths, pad_token",
                         [(batch_3, batch_padded_3, sequence_lengths_3, CHAR_FIELD,
                           None, None, None, None,
                           (lambda s: s.split()), True, "<wpad>")])
def test_nested_field(batch,
                      batch_padded,
                      sequence_lengths,
                      nesting_field,
                      init_token,
                      eos_token,
                      preprocessing,
                      postprocessing,
                      tokenize,
                      include_lengths,
                      pad_token):
  nested_field = NestedField(nesting_field=nesting_field,
                             init_token=init_token,
                             eos_token=eos_token,
                             preprocessing=preprocessing,
                             postprocessing=postprocessing,
                             tokenize=tokenize,
                             include_lengths=include_lengths,
                             pad_token=pad_token)
  batch, char_lengths = nested_field.process(batch)
  assert batch == batch_padded
  assert char_lengths == sequence_lengths


@pytest.mark.parametrize("batch, field, batch_padded, sequence_lengths",
                         [(batch_4, SENT_FIELD, batch_padded_4, sequence_lengths_4)])
def test_tri_nested_field(batch, field, batch_padded, sequence_lengths):
  padded, char_lengths = field.process(batch)
  assert padded == batch_padded
  assert sequence_lengths == char_lengths
