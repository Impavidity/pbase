from tqdm import tqdm
from collections import defaultdict
from functools import reduce
import array
import torch
import six
import numpy as np
import os

class EmbeddingFilter():
    """
    The text source should be separated with tab
    """
    def __init__(self, sourceEmbeddingPath= None):
        if sourceEmbeddingPath == None:
            print("Please Specify the source embedding path")
            return
        self.sourceEmbeddingPath = sourceEmbeddingPath
        self.dim = 0
        self.Embedding = self.loadEmbedding()
        self.Corpus = {}
        self.Vocab = {}
        self.Average = {}


    def toEmbedding(self, lower=True):
        if self.targetEmbeddingPath == None:
            print("Please Specify Target Embedding Path")
            return
        else:
            print("You are writing Embedding.txt to {}".format(self.targetEmbeddingPath))
        fout = open(self.targetEmbeddingPath, "w")
        processed = set([])
        totalVocab = reduce(lambda x, y: x| y, [self.Corpus[key] for key in self.Corpus.keys()])
        for token, embed in self.Embedding.items():
            if lower:
                token = token.lower()
            if token in totalVocab and token not in processed:
                fout.write(" ".join([token] + [str(x) for x in embed])+"\n")
                processed.add(token)
        fout.close()

    def toBinary(self, lower=True):
        if self.targetBinaryPath == None:
            print("Please Specify Target Binary Path")
            return
        else:
            print("You are writing Embedding.pt to {}".format(self.targetBinaryPath))
        itos, vectors, dim = [], array.array('d'), None
        self.toEmbedding(lower=lower)
        with open(self.targetEmbeddingPath, 'rb') as handler:
            lines = [line for line in handler]
        print("Loading vectors from {}".format(self.targetEmbeddingPath))
        for line in tqdm(lines, total=len(lines)):
            entries = line.strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if dim is None:
                dim = len(entries)
                self.dim = dim
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignore')
                continue
            vectors.extend(float(x) for x in entries)
            itos.append(word)
        stoi = {word: i for i, word in enumerate(itos)}
        vectors = torch.Tensor(vectors).view(-1, dim)
        print('saving vector to', self.targetBinaryPath)
        torch.save((stoi, vectors, dim), self.targetBinaryPath)

    def extractAverageEmbedding(self, corpusName):
        if corpusName not in self.Corpus:
            print("{} does not exist".format(corpusName))
            return None
        sentences = self.Corpus[corpusName]
        ave_embedding = []
        for sentence in tqdm(sentences):
            feature_sum = np.zeros(self.dim)
            count = 0
            for token in sentence:
                if token in self.Embedding:
                    feature_sum += np.array(self.Embedding[token])
                    count += 1
            if count != 0:
                feature_ave = feature_sum / count
            else:
                feature_ave = feature_sum
            ave_embedding.append(feature_ave)
        self.Average[corpusName] = ave_embedding
        return ave_embedding


    def addCorpus(self, corpusName, sourceTextPath, textIndex, delimiter='\t'):
        sentences = []
        words = set([])
        fin = open(sourceTextPath)
        for line in tqdm(fin.readlines()):
            items = line.strip().split(delimiter)
            sentence = items[textIndex]
            tokens = sentence.strip().split()
            sentences.append(tokens)
            words |= set(tokens)
        self.Corpus[corpusName] = sentences
        self.Corpus[corpusName] = words

    def loadEmbedding(self):
        fin = open(self.sourceEmbeddingPath)
        d = defaultdict(list)
        for line in tqdm(fin.readlines()):
            line = line.strip().split()
            if len(line) == 2:
                continue
            try:
                d[line[0]] = [float(x) for x in line[1:]]
            except:
                continue
        return d

    def setEmbeddingPath(self, sourceEmbeddingPath):
        self.sourceEmbeddingPath = sourceEmbeddingPath
        self.Embedding = self.loadEmbedding()

    def setTargetEmbeddingPath(self, targetEmbeddingPath):
        self.targetEmbeddingPath = targetEmbeddingPath

    def setTargetBinaryPath(self, targetBinaryPath):
        self.targetBinaryPath = targetBinaryPath

    def clearCorpus(self, corpusName):
        self.Corpus.pop(corpusName, None)
        self.Vocab.pop(corpusName, None)

    def clearAll(self):
        for key in self.Corpus.keys():
            self.clearCorpus(key)