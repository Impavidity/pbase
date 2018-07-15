from pbase.ptext.example import Example
from pbase.ptext.attribute import Attribute
from pbase.ptext.field import Field, NestedField

import pytest


EXAMPLE_DICT_1 = {
  "text": "Mary loves John",
  "label": 1,
  "feature": [3.3, 5.2]
}
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
VALUE_WORD_1 = ["mary", "loves", "john"]
VALUE_CHAR_1 = [['m', 'a', 'r', 'y'],
                ['l', 'o', 'v', 'e', 's'],
                ['j', 'o', 'h', 'n']]
VALUE_LABEL_1 = 1
VALUE_FEATURE_1 = [3.3, 5.2]
VALUE_LIST_1 = [VALUE_WORD_1,
                VALUE_CHAR_1,
                VALUE_LABEL_1,
                VALUE_FEATURE_1]


@pytest.mark.parametrize("example_dict, attributes, value_list",
                         [(EXAMPLE_DICT_1, ATTRIBUTES_1, VALUE_LIST_1)])
def test_example(example_dict, attributes, value_list):
  example = Example(example_dict=example_dict,
                    attributes=attributes)
  for attribute, value in zip(attributes, value_list):
    assert getattr(example, attribute.target) == value