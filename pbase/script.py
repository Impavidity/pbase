from tqdm import tqdm
from collections import defaultdict
from functools import reduce
from itertools import product
import subprocess
from random import randint
import array
import torch
import six
import os
import subprocess
import numpy as np
from six.moves.urllib.request import urlretrieve
import zipfile
from .utils import reporthook
import random
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


class EmbeddingFilter:
    """
    The text source should be separated with tab
    """
    def __init__(self, sourceEmbeddingPath= None):
        if sourceEmbeddingPath == None:
            print("You are not specify any Embedding Path")
            return
        self.sourceEmbeddingPath = sourceEmbeddingPath
        self.dim = 0
        self.Embedding = self.loadEmbedding()
        self.Corpus = {}
        self.Vocab = {}
        self.Average = {}


    def toEmbedding(self, lower=True, stem=False):
        if self.targetEmbeddingPath == None:
            print("Please Specify Target Embedding Path")
            return
        else:
            print("You are writing Embedding.txt to {}".format(self.targetEmbeddingPath))
        fout = open(self.targetEmbeddingPath, "w")
        processed = set([])
        totalVocab = reduce(lambda x, y: x| y, [self.Vocab[key] for key in self.Vocab.keys()])
        print("Total Vocab : {}".format(len(totalVocab)))
        for token, embed in self.Embedding.items():
            if lower:
                token = token.lower()
            if stem:
                token = stemmer.stem(token)
            if token in totalVocab and token not in processed:
                fout.write(" ".join([token] + [str(x) for x in embed])+"\n")
                processed.add(token)
        fout.close()

    def toBinary(self, lower=True, toEmbedFirst=True, stem=False):
        if self.targetBinaryPath == None:
            print("Please Specify Target Binary Path")
            return
        else:
            print("You are writing Embedding.pt to {}".format(self.targetBinaryPath))
        itos, vectors, dim = [], array.array('d'), None
        if toEmbedFirst:
            self.toEmbedding(lower=lower, stem=stem)
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
        self.Vocab[corpusName] = words
        print("Add {} sentences, and {} words".format(len(sentences), len(words)))

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
        for key in list(self.Corpus.keys()):
            self.clearCorpus(key)

class LinguisticFeatureAnnotator:
    """
    Need to download stanford Core NLP
    one sentence one line, tokenized beforehand
    props:
        ssplit.eolonly=true
        tokenize.whitespace=true
    """
    def __init__(self, stanfordCoreNLPPath=None):
        self.stanfordCoreNLPPath = stanfordCoreNLPPath
        self.url = "http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip"
        self.caseless_model_url = "http://nlp.stanford.edu/software/stanford-english-corenlp-2017-06-09-models.jar"
        self.dir = "stanford-corenlp-full-2017-06-09"
        self.tmp = "/tmp/annotator"
        os.makedirs(self.tmp, exist_ok=True)
        self.corpusList = {}
        self.prop_path = None
        if self.stanfordCoreNLPPath != None:
            self.prop_path = os.path.abspath(os.path.join(self.stanfordCoreNLPPath, self.dir, 'prop'))
            fout = open(self.prop_path, 'w')
            fout.write("{}\n{}".format("ssplit.eolonly=true", "tokenize.whitespace=true"))
            fout.close()

    def setStanfordCoreNLPPath(self, stanfordCoreNLPPath):
        self.stanfordCoreNLPPath = stanfordCoreNLPPath

    """
    CoreNLP version 3.8.0
    """
    def downloadStanfordCoreNLP(self, stanfordCoreNLPPath):
        self.stanfordCoreNLPPath = stanfordCoreNLPPath
        dest = os.path.join(self.stanfordCoreNLPPath, os.path.basename(self.url))
        if not os.path.isfile(dest):
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                urlretrieve(self.url, dest, reporthook=reporthook(t))
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(self.stanfordCoreNLPPath)
        self.prop_path = os.path.abspath(os.path.join(self.stanfordCoreNLPPath, self.dir, 'prop'))
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(self.caseless_model_url)) as t:
            urlretrieve(self.caseless_model_url, os.path.join(self.stanfordCoreNLPPath, self.dir,
                        os.path.basename(self.caseless_model_url)),reporthook=reporthook(t))
        fout = open(self.prop_path, 'w')
        fout.write("{}\n{}".format("ssplit.eolonly=true", "tokenize.whitespace=true"))
        fout.close()


    def addCorpus(self, corpusName, path, textIndex, dilimiter='\t'):
        fin = open(path)
        dest = os.path.join(self.tmp, corpusName)
        fout = open(dest, 'w')
        for line in fin.readlines():
            text = line.strip().split(dilimiter)[textIndex]
            fout.write(text+"\n")
        fout.close()
        self.corpusList[corpusName] = path

    def annotate(self, corpusName, output, anno_type="normal", tag_head_word=False, tag_head_lamma=False, tag_head_pos=False):
        if corpusName not in self.corpusList:
            print("Please add corpus {} before annotating".format(corpusName))
            return
        if anno_type == "normal":
            subprocess.run(['java', '-XX:-UseGCOverheadLimit', '-cp',
                            '{}/*'.format(os.path.abspath(os.path.join(self.stanfordCoreNLPPath, self.dir))),
                            '-Xmx3g', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                            '-annotators', 'tokenize,ssplit,pos,lemma,ner,depparse',
                            '-file', os.path.join(self.tmp, corpusName),
                            '-props', self.prop_path,
                            '-outputFormat', 'conll',
                            '-outputDirectory', self.tmp])
        elif anno_type == "caseless":
            subprocess.run(['java', '-XX:-UseGCOverheadLimit', '-cp',
                            '{}/*'.format(os.path.abspath(os.path.join(self.stanfordCoreNLPPath, self.dir))),
                            '-Xmx3g', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                            '-annotators', 'tokenize,ssplit,pos,lemma, ner, parse',
                            '-file', os.path.join(self.tmp, corpusName),
                            '-props', self.prop_path,
                            '-outputFormat', 'conll',
                            '-pos.model', 'edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger',
                            '-ner.model', 'edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz,'\
                            'edu/stanford/nlp/models/ner/english.muc.7class.caseless.distsim.crf.ser.gz,'\
                            'edu/stanford/nlp/models/ner/english.conll.4class.caseless.distsim.crf.ser.gz',
                            '-parse.model', 'edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz',
                            '-outputDirectory', self.tmp])
        ftext = open(self.corpusList[corpusName])
        fanno = open(os.path.join(self.tmp, corpusName+'.conll'))
        fout = open(output, 'w')
        # Extract features on CONLL format
        sent_num = 0
        words, lamma, pos, ner, dep = [], [], [], [], []
        head_word, head_lamma, head_pos = [], [], []
        tokens = []
        linguisticFeature = []
        for line in fanno.readlines():
            items = line.strip().split('\t')
            if len(items) == 1: # len(['']) == 1
                for token in tokens:
                    words.append(token[1])
                    lamma.append(token[2])
                    pos.append(token[3])
                    ner.append(token[4])
                    dep.append(token[6])
                    if tag_head_word:
                        head_word.append(tokens[int(token[5]) - 1][1] if int(token[5]) != 0 else 'ROOT')
                    if tag_head_lamma:
                        head_lamma.append(tokens[int(token[5]) - 1][1] if int(token[5]) != 0 else 'ROOT')
                    if tag_head_pos:
                        head_pos.append(tokens[int(token[5]) - 1][1] if int(token[5]) != 0 else 'ROOT')
                lamma_feature, pos_feature, dep_feature, ner_feature = " ".join(lamma), " ".join(pos), " ".join(dep), " ".join(ner)
                sentence_feature = [lamma_feature, pos_feature, dep_feature, ner_feature]
                if tag_head_word:
                    sentence_feature.append(head_word)
                if tag_head_lamma:
                    sentence_feature.append(head_lamma)
                if tag_head_pos:
                    sentence_feature.append(head_pos)
                linguisticFeature.append("\t".join(sentence_feature))
                sent_num += 1
                words, lamma, pos, ner, dep = [], [], [], [], []
                head_word, head_lamma, head_pos = [], [], []
                tokens = []
            else:
                tokens.append(items)
        check_sent_num = 0
        for line_id, line in enumerate(ftext.readlines()):
            line = line.strip()
            fout.write("{}\t{}\n".format(line, linguisticFeature[line_id]))
            check_sent_num += 1
        if (check_sent_num != sent_num):
            print("Sentences Number Mismatch in Converting CONLL to corpus")
            exit()

class RandomTrainer:
    def __init__(self, script, random_seed_arg, model_prefix_arg, save_path_arg, log_dir, model_dir, round_num):
        self.script = script
        self.random_seed_arg = random_seed_arg
        self.model_prefix_arg = model_prefix_arg
        self.save_path_arg = save_path_arg
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.round_num = round_num
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    def start(self, process_num=1):
        process_list = []
        for i in range(self.round_num):
            gen_random = randint(0, 65535)
            print(i, gen_random)
            with open(os.path.join(self.log_dir, "{}_{}.log".format(i, gen_random)), "w") as handler:
                if len(process_list) >= process_num:
                    exit_codes = [p.wait() for p in process_list]
                    process_list = []
                proc = subprocess.Popen(
                    self.script.split() +
                    [
                        self.random_seed_arg, str(gen_random),
                        self.model_prefix_arg, "{}_{}".format(i, gen_random),
                        self.save_path_arg, self.model_dir
                    ], stdout=handler)
                process_list.append(proc)


class RandomTester:
    def __init__(self, log_dir):
        self.pipeline = []
        self.ranges = []
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def add_pipeline(self, script, model_arg, result_parser, log_dir, model_dir, ignore_last, limit=None):
        self.pipeline.append((script, model_arg, result_parser, log_dir, model_dir, ignore_last, limit))

    def add_specific_step(self, script):
        self.pipeline.append((script,))

    def start(self):
        for x in self.pipeline:
            if len(x) == 1:
                self.ranges.append([x[0]])
            else:
                model_lists = sorted(filter(lambda f: f.endswith('pt'), os.listdir(x[4])),
                                   key=lambda f:int(f.split('_')[0]))
                if x[-2]:
                    model_lists = model_lists[:-1]
                if x[-1] is not None:
                    model_lists = model_lists[:x[-1]]
                scripts = []
                for model_file in model_lists:
                    script = "{} {} {}".format(x[0], x[1], os.path.join(x[4], model_file))
                    scripts.append(script)
                self.ranges.append(scripts)
        for idx, pipeline in tqdm(enumerate(product(*self.ranges))):
            # Simple version of pipeline
            # TODO: make it save time
            with open(os.path.join(self.log_dir, "{}.log".format(idx)), 'a+') as handler:
                for cmd in pipeline:
                    subprocess.run(cmd.split(), stdout=handler)

    def parse_results(self, parser):
        results = []
        for file in os.listdir(self.log_dir):
            if file.endswith('log'):
                text = open(os.path.join(self.log_dir, file)).readlines()
                results.append(parser(text))
        return results

class PosNegPairGenerator:
    def __init__(self):
        pass

    def convertCorpus(self, corpus_name, input_path, group_index, label_index, pos_label, neg_label, output_path, neg_sample_size=None):
        fin = open(input_path)
        fout = open(output_path, 'w')
        corpus = defaultdict(lambda : defaultdict(list))
        for line in fin.readlines():
            items = line.strip().split('\t')
            group_by = items[group_index]
            label = items[label_index]
            corpus[group_by][label].append(line.strip())
        for group in corpus.keys():
            if len(corpus[group]) > 2:
                print("Only allow 2 labels")
                print(corpus[group].keys())
                return
            pos = corpus[group][pos_label]
            neg = corpus[group][neg_label]
            for pos_item in pos:
                if neg_sample_size != None:
                    neg = random.sample(neg, neg_sample_size)
                for neg_item in neg:
                    fout.write("{}\t{}\n".format(pos_item, neg_item))
        fout.close()



    def convertCorpusWithRanking(self, corpus_name, input_path, group_index, label_index,
                             ranking_index, pos_label, neg_label, output_path):
        pass