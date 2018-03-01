from collections import defaultdict
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm


class PairwiseFeature:
    def __init__(self):
        self.corpus = defaultdict(list)
        self.word_cnt = defaultdict(lambda : defaultdict(int))

    def addCorpus(self, corpusName, path, textIndex1, textIndex2, dilimiter='\t'):
        fin = open(path)
        for line in tqdm(fin.readlines()):
            items = line.strip().split(dilimiter)
            sentence1 = items[textIndex1]
            sentence2 = items[textIndex2]
            self.corpus[corpusName].append((sentence1, sentence2))
        fin.close()

    def overlapFeature(self, corpusName, dumpPath=None):
        """
        Step1 : Get pairwise word to document frequency.
        For index i, if sentence i in sent_list_1 and sentence i in sent_list_2 both
        container word w, then w is counted only once.
        Returns a dictionary mapping words to number of sentence pairs the word appears in.
        Step2 : Get overlap, idf weighted overlap,
        overlap excluding stopwords, and idf weighted overlap excluding stopwords.
        """
        sent_list1 = []
        sent_list2 = []
        for s1, s2 in self.corpus[corpusName]:
            sent1 = s1.rstrip('.').split(' ')
            sent2 = s2.rstrip('.').split(' ')
            sent_list1.append(sent1)
            sent_list2.append(sent2)
            unique_tokens = set(sent1) | set(sent2)
            for t in unique_tokens:
                self.word_cnt[corpusName][t] += 1
        stoplist = set(stopwords.words('english'))
        num_docs = len(self.corpus[corpusName])
        overlap_feats = []
        for s1, s2 in zip(sent_list1, sent_list2):
            tokens_a_set, tokens_b_set = set(s1), set(s2)
            intersect = tokens_a_set & tokens_b_set
            overlap = len(intersect) / (len(tokens_a_set) + len(tokens_b_set))
            idf_intersect = sum(np.math.log(num_docs / self.word_cnt[corpusName][w]) for w in intersect)
            idf_weighted_overlap = idf_intersect / (len(tokens_a_set) + len(tokens_b_set))

            tokens_a_set_no_stop = set(w for w in s1 if w not in stoplist)
            tokens_b_set_no_stop = set(w for w in s2 if w not in stoplist)
            intersect_no_stop = tokens_a_set_no_stop & tokens_b_set_no_stop
            overlap_no_stop = len(intersect_no_stop) / (len(tokens_a_set_no_stop) + len(tokens_b_set_no_stop))
            idf_intersect_no_stop = sum(np.math.log(num_docs / self.word_cnt[corpusName][w]) for w in intersect_no_stop)
            idf_weighted_overlap_no_stop = idf_intersect_no_stop / (len(tokens_a_set_no_stop) + len(tokens_b_set_no_stop))
            overlap_feats.append([overlap, idf_weighted_overlap, overlap_no_stop, idf_weighted_overlap_no_stop])
        if dumpPath != None:
            fout = open(dumpPath, 'w')
            for items in overlap_feats:
                fout.write(" ".join([str(item) for item in items])+"\n")
            fout.close()
        return overlap_feats

    def intersectFeature(self, corpusName, dumpPath=None, twitter=False):
        sent_list1 = []
        sent_list2 = []
        for s1, s2 in self.corpus[corpusName]:
            sent1 = s1.rstrip('.').split()
            sent2 = s2.rstrip('.').split()
            sent_list1.append(sent1)
            sent_list2.append(sent2)
            unique_tokens = set(sent1) | set(sent2)
            for t in unique_tokens:
                self.word_cnt[corpusName][t] += 1
        stoplist = set(stopwords.words('english'))
        intersect_feats = []
        num_docs = len(self.corpus[corpusName])
        for s1, s2 in zip(sent_list1, sent_list2):
            tokens_a_set_no_stop = set(w for w in s1 if w not in stoplist)
            tokens_b_set_no_stop = set(w for w in s2 if w not in stoplist)
            twitter_intersect_no_stop = []
            for token_a in tokens_a_set_no_stop:
                for token_b in tokens_b_set_no_stop:
                    if (token_a in token_b or token_b in token_a) and (abs(len(token_a)-len(token_b)) <= 1):
                        twitter_intersect_no_stop.append(token_a)
            twitter_feature = len(list(set(twitter_intersect_no_stop))) / len(tokens_a_set_no_stop)
            intersect_no_stop = tokens_a_set_no_stop & tokens_b_set_no_stop
            overlap_no_stop = len(intersect_no_stop) / len(tokens_a_set_no_stop)
            idf_intersect_no_stop = sum(np.math.log(num_docs / self.word_cnt[corpusName][w]) for w in intersect_no_stop)
            idf_overlap_no_stop = idf_intersect_no_stop / len(tokens_a_set_no_stop)
            if twitter:
                intersect_feats.append([overlap_no_stop, idf_overlap_no_stop, twitter_feature])
            else:
                intersect_feats.append([overlap_no_stop, idf_overlap_no_stop])
        if dumpPath != None:
            fout = open(dumpPath, 'w')
            for items in intersect_feats:
                fout.write(" ".join([str(item) for item in items])+"\n")
            fout.close()
        return intersect_feats








