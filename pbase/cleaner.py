import unicodedata
import re
import codecs
from nltk.tokenize.treebank import TreebankWordTokenizer

# Normalize text by mapping non-ascii characters to approximate ascii. e.g., beyonc'{e} becomes beyonce
def normalize_unicode(text):
  #return text.encode('ascii', 'ignore')
  return unicodedata.normalize('NFD', text).encode('ascii', 'ignore')

# Standard word tokenizer.
_treebank_word_tokenize = TreebankWordTokenizer().tokenize

def word_tokenize(text, language='english'):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).
    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    return [token for token in _treebank_word_tokenize(text)]


def clean_text(text):
  """
  Prepare question text for tokenization: lowercase, remove punctuation, and remove episode numbers (these are added during Spark pipeline)
  e.g., "Who plays in Seinfeld: The Contest S10E8?" ==> "who plays in seinfeld the contest"
  :param question: string representing question (not tokenized)
  :return: string representing cleaned up question, ready for tokenization
  """
  question = re.sub("[\.\t\,\:;\(\)\?\!]", " ", text.lower(), 0, 0)
  return re.sub("s\d+e\d+", "", question, 0, 0)

