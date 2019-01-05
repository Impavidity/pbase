import logging
import os
from collections import defaultdict

from nltk import Tree

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OntonotesSentence(object):
  """
  A class representing the annotations available for a single CONLL formatted sentence.

  Args:
    document_id (str): document filename
    sentence_id (int): the integer ID of the sentence within a document
    words (list(str)): tokens as segmented/tokenized in the Treebank
    pos_tags (list(str)):
    parse_tree:
    predicate_lemmas:
    predicate_framenet_ids:
    word_senses:
    speakers:
    named_entities:
    srl_frames:
    coref_spans:
  """

  def __init__(self, document_id, sentence_id, words, pos_tags, parse_tree,
               predicate_lemmas, predicate_framenet_ids, word_senses, speakers,
               named_entities, srl_frames, coref_spans):
    self.document_id = document_id
    self.sentence_id = sentence_id
    self.words = words
    self.pos_tags = pos_tags
    self.parse_tree = parse_tree
    self.predicate_lemmas = predicate_lemmas
    self.predicate_framenet_ids = predicate_framenet_ids
    self.word_senses = word_senses
    self.speakers = speakers
    self.named_entities = named_entities
    self.srl_frames = srl_frames
    self.coref_spans = coref_spans


class Ontonotes(object):
  def dataset_iterator(self, file_path):
    """

    Args:
      file_path (str): The path to the train/valid/test dataset

    Returns: (iterator(str))
    """
    for conll_file in self.dataset_path_iterator(file_path):
      yield from self.sentence_iterator(conll_file)

  @staticmethod
  def dataset_path_iterator(file_path):
    """

    Args:
      file_path:

    Returns:

    """
    logger.info("Reading CONLL sentences from dataset files at: %s", file_path)
    for root, _, files in list(os.walk(file_path)):
      for data_file in files:
        # These are a relic of the dataset pre-processing. Every
        # file will be duplicated - one file called filename.gold_skel
        # and one generated from the preprocessing called filename.gold_conll.
        if not data_file.endswith("gold_conll"):
          continue
        yield os.path.join(root, data_file)

  def sentence_iterator(self, file_path):
    """

    Args:
      file_path:

    Returns:

    """
    for document in self.dataset_document_iterator(file_path):
      for sentence in document:
        yield sentence

  def dataset_document_iterator(self, file_path):
    """

    Args:
      file_path:

    Returns:

    """
    with open(file_path, 'r', encoding='utf8') as open_file:
      conll_rows = []
      document = []
      for line in open_file:
        line = line.strip()
        if line != '' and not line.startswith('#'):
          # Non-empty line. Collect the annotation.
          conll_rows.append(line)
        else:
          if conll_rows:
            document.append(self._conll_rows_to_sentence(conll_rows))
            conll_rows = []
        if line.startswith("#end document"):
          yield document
          document = []
      if document:
        # Collect any stragglers or files which might not
        # have the '#end document' format for the end of the file.
        yield document

  def _conll_rows_to_sentence(self, conll_rows):
    """

    Args:
      conll_rows:

    Returns:

    """
    document_id = None
    sentence_id = None
    # The words in the sentence.
    sentence = []
    # The pos tags of the words in the sentence.
    pos_tags = []
    # the pieces of the parse tree.
    parse_pieces = []
    # The lemmatised form of the words in the sentence which
    # have SRL or word sense information.
    predicate_lemmas = []
    # The FrameNet ID of the predicate.
    predicate_framenet_ids = []
    # The sense of the word, if available.
    word_senses = []
    # The current speaker, if available.
    speakers = []

    verbal_predicates = []
    span_labels = []
    current_span_labels = []

    # Cluster id -> List of (start_index, end_index) spans.
    clusters = defaultdict(list)
    # Cluster id -> List of start_indices which are open for this id.
    coref_stacks = defaultdict(list)

    for index, row in enumerate(conll_rows):
      conll_components = row.split()

      document_id = conll_components[0]
      sentence_id = int(conll_components[1])
      word = conll_components[3]
      pos_tag = conll_components[4]
      parse_piece = conll_components[5]

      # Replace brackets in text and pos tags
      # with a different token for parse trees.
      if pos_tag != "XX" and word != "XX":
        if word == "(":
          parse_word = "-LRB-"
        elif word == ")":
          parse_word = "-RRB-"
        else:
          parse_word = word
        if pos_tag == '(':
          pos_tag = '-LRB-'
        if pos_tag == ')':
          pos_tag = '-RRB-'
        (left_brackets, right_hand_side) = parse_piece.split('*')
        # only keep ')' if there are nested brackets with nothing in them.
        right_brackets = right_hand_side.count(')') * ')'
        parse_piece = f'{left_brackets} ({pos_tag} {parse_word}) {right_brackets}'
      else:
        # There are some bad annotations in the CONLL data.
        # They contain no information, so to make this explicit,
        # we just set the parse piece to be None which will result
        # in the overall parse tree being None.
        parse_piece = None

      lemmatised_word = conll_components[6]
      framenet_id = conll_components[7]
      word_sense = conll_components[8]
      speaker = conll_components[9]

      if not span_labels:
        # If this is the first word in the sentence, create
        # empty lists to collect the NER and SRL BIO labels.
        # We can't do this upfront, because we don't know how many
        # components we are collecting, as a sentence can have
        # variable numbers of SRL frames.
        span_labels = [[] for _ in conll_components[10:-1]]
        # Create variables representing the current label for each label
        # sequence we are collecting.
        current_span_labels = [None for _ in conll_components[10:-1]]

      self._process_span_annotations_for_word(conll_components[10:-1],
                                              span_labels, current_span_labels)

      # If any annotation marks this word as a verb predicate,
      # we need to record its index. This also has the side effect
      # of ordering the verbal predicates by their location in the
      # sentence, automatically aligning them with the annotations.
      word_is_verbal_predicate = any(
          ["(V" in x for x in conll_components[11:-1]])
      if word_is_verbal_predicate:
        verbal_predicates.append(word)

      self._process_coref_span_annotations_for_word(
          conll_components[-1], index, clusters, coref_stacks)

      sentence.append(word)
      pos_tags.append(pos_tag)
      parse_pieces.append(parse_piece)
      predicate_lemmas.append(
          lemmatised_word if lemmatised_word != "-" else None)
      predicate_framenet_ids.append(
          framenet_id if framenet_id != "-" else None)
      word_senses.append(float(word_sense) if word_sense != "-" else None)
      speakers.append(speaker if speaker != "-" else None)

    named_entities = span_labels[0]
    srl_frames = [
        (predicate, labels)
        for predicate, labels in zip(verbal_predicates, span_labels[1:])
    ]

    if all(parse_pieces):
      parse_tree = Tree.fromstring("".join(parse_pieces))
    else:
      parse_tree = None
    coref_span_tuples = {(cluster_id, span)
                         for cluster_id, span_list in clusters.items()
                         for span in span_list}
    return OntonotesSentence(document_id, sentence_id, sentence, pos_tags,
                             parse_tree, predicate_lemmas,
                             predicate_framenet_ids, word_senses, speakers,
                             named_entities, srl_frames, coref_span_tuples)

  @staticmethod
  def _process_coref_span_annotations_for_word(label, word_index, clusters,
                                               coref_stacks):
    """

    Args:
      label:
      word_index:
      clusters:
      coref_stacks:

    Returns:

    """
    if label != "-":
      for segment in label.split("|"):
        # The conll representation of coref spans allows spans to
        # overlap. If spans end or begin at the same word, they are
        # separated by a "|".
        if segment[0] == "(":
          # The span begins at this word.
          if segment[-1] == ")":
            # The span begins and ends at this word (single word span).
            cluster_id = int(segment[1:-1])
            clusters[cluster_id].append((word_index, word_index))
          else:
            # The span is starting, so we record the index of the word.
            cluster_id = int(segment[1:])
            coref_stacks[cluster_id].append(word_index)
        else:
          # The span for this id is ending, but didn't start at this word.
          # Retrieve the start index from the document state and
          # add the span to the clusters for this id.
          cluster_id = int(segment[:-1])
          start = coref_stacks[cluster_id].pop()
          clusters[cluster_id].append((start, word_index))

  @staticmethod
  def _process_span_annotations_for_word(annotations, span_labels,
                                         current_span_labels):
    """

    Args:
      annotations:
      span_labels:
      current_span_labels:

    Returns:

    """
    for annotation_index, annotation in enumerate(annotations):
      # strip all bracketing information to
      # get the actual propbank label.
      label = annotation.strip("()*")

      if "(" in annotation:
        # Entering into a span for a particular semantic role label.
        # We append the label and set the current span for this annotation.
        bio_label = "B-" + label
        span_labels[annotation_index].append(bio_label)
        current_span_labels[annotation_index] = label
      elif current_span_labels[annotation_index] is not None:
        # If there's no '(' token, but the current_span_label is not None,
        # then we are inside a span.
        bio_label = "I-" + current_span_labels[annotation_index]
        span_labels[annotation_index].append(bio_label)
      else:
        # We're outside a span.
        span_labels[annotation_index].append("O")
      # Exiting a span, so we reset the current span label for this annotation.
      if ")" in annotation:
        current_span_labels[annotation_index] = None
