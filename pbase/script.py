from tqdm import tqdm
from collections import defaultdict
from functools import reduce

class EmbeddingFilter():
    """
    The text source should be separated with tab
    """
    def __init__(self, sourceEmbeddingPath= None, targetEmbeddingPath=None, targetBinaryPath=None):
        if sourceEmbeddingPath == None:
            print("Please Specify the source embedding path")
            return
        self.sourceEmbeddingPath = sourceEmbeddingPath
        self.targetEmbeddingPath = targetEmbeddingPath
        self.targetBinaryPath = targetBinaryPath
        self.Embedding = self.loadEmbedding()
        self.Corpus = {}
        self.Vocab = {}

    def toEmbedding(self, lower=True):
        if self.targetEmbeddingPath == None:
            print("Please Specify Target Embedding Path")
            return
        fout = open(self.targetEmbeddingPath, "w")
        processed = set([])
        totalVocab = reduce(lambda x, y: x| y, [self.Corpus[key] for key in self.Corpus.keys()])
        for token, embed in self.Embedding.items():
            if lower:
                token = token.lower()
            if token in totalVocab and token not in processed:
                embed = self.Embedding[token]
                fout.write(" ".join([token] + embed))
                processed.add(token)

    def toBinary(self):
        pass

    def extractSumEmbeding(self):
        return

    def extractAverageEmbedding(self):
        return

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
        for line in tqdm(enumerate(fin.readlines())):
            line = line.strip().split()
            if len(line) == 2:
                continue
            d[line[0]] = [float(x) for x in line[1:]]
        return d

    def setEmbeddingPath(self, sourceEmbeddingPath):
        self.sourceEmbeddingPath = sourceEmbeddingPath
        self.Embedding = self.loadEmbedding()

    def setTargetEmbeddingPath(self, targetEmbeddingPath):
        self.targetEmbeddingPath = targetEmbeddingPath

    def clearCorpus(self, corpusName):
        self.Corpus.pop(corpusName, None)
        self.Vocab.pop(corpusName, None)

    def clearAll(self):
        for key in self.Corpus.keys():
            self.clearCorpus(key)