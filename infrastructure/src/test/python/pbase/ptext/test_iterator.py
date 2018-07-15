from pbase.ptext.iterator import Iterator
from pbase.ptext.attribute import Attribute
from pbase.ptext.field import Field, NestedField
from pbase.ptext.dataset import Dataset

import pytest


EXAMPLE_DICT_1 = {
  "text": "Mary loves John",
  "label": '1',
  "feature": [3.3, 5.2]
}
EXAMPLE_DICT_2 = {
  "text": "John cries",
  "label": '0',
  "feature": [4.3, 2.2]
}
EXAMPLE_DICT_3 = {
  "text": "Will you marry me ?",
  "label": '1',
  "feature": [0.3, 0.2]
}
EXAMPLES_1 = [EXAMPLE_DICT_1, EXAMPLE_DICT_2, EXAMPLE_DICT_3]
WORD_FIELD = Field(sequential=True,
                   lower=True,
                   include_lengths=True)
CHAR_FIELD = NestedField(Field(sequential=True,
                               lower=True,
                               tokenize=list,
                               pad_token='<c>',
                               include_lengths=True),
                         include_lengths=True)
LABEL_FIELD = Field(sequential=False)
FEATURE_FIELD = Field(sequential=False)
ATTRIBUTES_1 = [Attribute("word", "text", WORD_FIELD),
                Attribute("char", "text", CHAR_FIELD),
                Attribute("label", "label", LABEL_FIELD),
                Attribute("feature", "feature", FEATURE_FIELD)]
BATCH_LENGTHS_1 = [2, 1]


@pytest.mark.parametrize("examples, attributes, batch_lengths",
                         [(EXAMPLES_1, ATTRIBUTES_1, BATCH_LENGTHS_1)])
def test_iterator(examples, attributes, batch_lengths):
  dataset = Dataset(examples=examples, attributes=attributes)
  data_iterator = Iterator(dataset=dataset, batch_size=2, shuffle=False)
  lengths = []
  for batch_idx, batch in enumerate(data_iterator):
    lengths.append(len(batch))
  assert lengths == batch_lengths