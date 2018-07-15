from pbase.ptext.pipeline import Pipeline

import pytest


convert_token1 = str.split
examples1 = [["hello world", "hello pbase"], ["I love pbase"]]
results1 = [[["hello", "world"], ["hello", "pbase"]], [["I", "love", "pbase"]]]

convert_token2 = str.upper
results2 = [["HELLO WORLD", "HELLO PBASE"], ["I LOVE PBASE"]]

results3 = [[["HELLO", "WORLD"], ["HELLO", "PBASE"]], [["I", "LOVE", "PBASE"]]]


@pytest.mark.parametrize("convert_tokens, examples, results",
                         [(convert_token1, examples1, results1),
                          (convert_token2, examples1, results2),
                          ([convert_token1, convert_token2], examples1, results3)])
def test_pipeline(convert_tokens, examples, results):
  if isinstance(convert_tokens, list):
    pipeline = Pipeline()
    for convert_token in convert_tokens:
      pipeline.add_after(pipeline=convert_token)
  else:
    pipeline = Pipeline(convert_token=convert_tokens)
  processed = pipeline(examples)
  assert processed == results