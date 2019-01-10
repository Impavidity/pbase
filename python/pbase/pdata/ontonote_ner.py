import json
import logging

from pbase.pdata.utils import Ontonotes, change_coding_schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OntonotesNamedEntityRecognition(object):
  """

  Args:
    domain_identifier (str):
    coding_scheme (str):
  """

  def __init__(self, domain_identifier=None, coding_scheme='BIO'):
    self.domain_identifier = domain_identifier
    self.coding_scheme = coding_scheme

  def read(self, file_path):
    logger.info("Reading Fine-Grained NER instance from dataset files at {}".
                format(file_path))
    ontonotes_reader = Ontonotes()
    if self.domain_identifier is not None:
      logger.info("Filtering to only include file paths containing {}".format(
          self.domain_identifier))
    for sentence in self._ontonotes_subset(ontonotes_reader, file_path,
                                           self.domain_identifier):
      yield sentence

  @staticmethod
  def _ontonotes_subset(ontonotes_reader, file_path, domain_identifier):
    """
    Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
    If the domain identifier is present, only examples which contain the domain
    identifier in the file path are yielded.
    """
    for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
      if (domain_identifier is None or
          f"/{domain_identifier}/" in conll_file) and "/pt/" not in conll_file:
        yield from ontonotes_reader.sentence_iterator(conll_file)

  def dump(self, file_path, dump_path):
    with open(dump_path, "w", encoding='utf-8') as fout:
      for sentence in self.read(file_path):
        tags = sentence.named_entities
        if self.coding_scheme != 'BIO':
          tags = change_coding_schema(
              sentence.named_entities,
              encoding='BIO',
              decoding=self.coding_scheme)
        fout.write(json.dumps({'tokens': sentence.words, 'tags': tags}) + "\n")
