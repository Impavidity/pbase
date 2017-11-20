from tqdm import tqdm
from fuzzywuzzy import process, fuzz
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
import logging


logger = logging.getLogger()
logger.disabled = True
tokenizer = TreebankWordTokenizer()

class Entity:
    def __init__(self):
        pass

    def MaskEntity(self, input, output, sent_id, tag_id,
                   delimiter='\t', tag_O='O', tag_I='I', pad_tag='<pad>', mask_tag='<e>'):
        fin = open(input)
        fout = open(output, "w")
        print("Processing {}".format(input))
        for line in tqdm(fin.readlines()):
            items = line.strip().split(delimiter)
            sentence = items[sent_id].strip().split()
            label = items[tag_id].strip().split()
            # if len(sentence) != len(label):
            #     print("Length mismatch in file : {}".format(args.input_file))
            sen_str = []
            e_str = []
            flag = False
            for token, tag in zip(sentence, label):
                if token == pad_tag:
                    break
                if tag == tag_O:
                    if flag:
                        flag = False
                    sen_str.append(token)
                if tag[0] == tag_I:
                    if flag == False:
                        sen_str.append(mask_tag)
                        flag = True
                    e_str.append(token)
            if len(e_str) == 0:
                # We regard the whole sentence as entity here
                fout.write("{}\t{}\n".format(" ".join(sen_str), " ".join(sen_str)))
            else:
                fout.write("{}\t{}\n".format(" ".join(sen_str), " ".join(e_str)))

    def get_indices(self, src_list, pattern_list):
        indices = None
        for i in range(len(src_list)):
            match = 1
            for j in range(len(pattern_list)):
                if src_list[i + j] != pattern_list[j]:
                    match = 0
                    break
            if match:
                indices = range(i, i + len(pattern_list))
                break
        return indices

    def get_ngram(self, tokens):
        ngram = []
        for i in range(1, len(tokens)+1):
            for s in range(len(tokens)-i+1):
                ngram.append((" ".join(tokens[s: s+i]), s, i+s))
        return ngram


    def ReverseLinking(self, sent, text_candidate):
        # Currently only IO scheme is supported
        # We assume that sent and text candidate are preprocessed and only need to be tokenized by space
        tokens = sent.split()
        label = ["O"] * len(tokens)
        text_attention_indices = None
        exact_match = False

        if text_candidate is None or len(text_candidate) == 0:
            return '<UNK>', label, exact_match

        # sorted by length
        for text in sorted(text_candidate, key=lambda x:len(x), reverse=True):
            pattern = r'(^|\s)(%s)($|\s)' % (re.escape(text))
            if re.search(pattern, sent):
                text_attention_indices = self.get_indices(tokens, text.split())
                break
        if text_attention_indices != None:
            exact_match = True
            for i in text_attention_indices:
                label[i] = 'I'
        else:
            try:
                v, score = process.extractOne(sent, text_candidate, scorer=fuzz.partial_ratio)
            except:
                print("Extraction Error with FuzzyWuzzy : {} || {}".format(sent, text_candidate))
                return '<UNK>', label, exact_match
            v = v.split()
            n_gram_candidate = self.get_ngram(tokens)
            n_gram_candidate = sorted(n_gram_candidate, key=lambda x: (fuzz.ratio(x[0], v)), reverse=True)
            top = n_gram_candidate[0]
            for i in range(top[1], top[2]):
                label[i] = 'I'
        entity_text = []
        for l, t in zip(label, tokens):
            if l == 'I':
                entity_text.append(t)
        entity_text = " ".join(entity_text)
        label = " ".join(label)
        return entity_text, label, exact_match





