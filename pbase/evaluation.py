from collections import Counter
import os
import subprocess

def get_span(label, with_type):
    span, tag = [], []
    st = -1
    en = -1
    flag = False
    for k, item in enumerate(label):
        if item[0] == 'I' and flag == False:
            flag = True
            st = k
            if with_type:
                tag.append(item[2:])
        if item[0] == 'I' and flag == True:
            if with_type:
                tag.append(item[2:])
        if item[0] != 'I' and flag == True:
            flag = False
            en = k
            if with_type:
                tag_counter = Counter(tag)
                max_tag = tag_counter.most_common()[0][0]
                span.append((st, en, max_tag))
            else:
                span.append((st, en))
            st = -1
            en = -1
    if st != -1 and en == -1:
        en = len(label)
        if with_type:
            tag_counter = Counter(tag)
            max_tag = tag_counter.most_common()[0][0]
            span.append((st, en, max_tag))
        else:
            span.append((st, en))
    return span


def NEREvaluation(goldLabel, predLabel, with_type=False):
    if len(goldLabel) != len(predLabel):
        print("Length is not matched {}/{}".format(len(goldLabel), len(predLabel)))
    right = 0
    pred_en, total_en = 0, 0
    for gold, pred in zip(goldLabel, predLabel):
        gold_span = get_span(gold, with_type)
        pred_span = get_span(pred, with_type)
        total_en += len(gold_span)
        pred_en += len(pred_span)
        for item in pred_span:
            if item in gold_span:
                right += 1
    if pred_en == 0:
        precision = 0
    else:
        precision = right / pred_en
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def MRR(pairs):
    """
    :param pairs:  A list of pair. One pair is a query. == (score, sim)
    :return: mean_reciprocal_rank
    """
    pass

def MAP(pairs):
    """
    :param pairs:  A list of pair. One pair is a query. == (score, sim)
    :return: mean_average_precision
    """
    pass

def precision_top_k(pairs, k):
    """
    :param pairs: A list of pair. One pair is a query. == (score, sim)
    :param k: precision on top k
    :return: P_k
    """
    pass

def TREC_MAP_MRR(pairs, pred_fname="pred.txt", gold_fname="gold.txt"):
    qrel_fname = gold_fname
    results_fname = pred_fname
    qrel_template = '{qid} 0 {docno} {rel}\n'
    results_template = "{qid} 0 {docno} 0 {sim} biaffine\n"
    with open(qrel_fname, 'w') as f1, open(results_fname, 'w') as f2:
        qid = 0
        for pair in pairs:
            qid += 1
            docno = 0
            for predicted, actual in zip(pair[0], pair[1]):
                docno += 1
                f1.write(qrel_template.format(qid=qid, docno=docno, rel=actual))
                f2.write(results_template.format(qid=qid, docno=docno, sim=predicted))

    trec_eval_path = '/mnt/collections/p8shi/dev/biaffine/qa/trec_eval-9.0.5/trec_eval'
    trec_out = subprocess.check_output([trec_eval_path, '-m', 'map', '-m', 'recip_rank', qrel_fname, results_fname])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[0].split('\t')[-1])
    mean_reciprocal_rank = float(trec_out_lines[1].split('\t')[-1])

    os.remove(qrel_fname)
    os.remove(results_fname)


    return mean_average_precision, mean_reciprocal_rank

def TWITTER_MAP_MRR(pairs, pred_fname="pred.txt", gold_fname=None, id_fname=None,
                    qid_index=None, docid_index=None, delimiter=' ', model="model"):
    if id_fname == None or gold_fname == None or qid_index == None or docid_index == None:
        print("You need to pass filename of qrel or qid/docid to the function")
        exit()
    qid_file = open(id_fname)
    id_list = []
    for line in qid_file.readlines():
        line = line.strip().split(delimiter)
        qid = line[qid_index]
        docid = line[docid_index]
        id_list.append((qid, docid))
    results_file = open(pred_fname, "w")
    results_template = "{qid} Q0 {docno} 0 {sim} {model}\n"
    counter = 0
    for pair in pairs:
        for predicted in pair[0]:
            qid = id_list[counter][0]
            docid = id_list[counter][1]
            results_file.write(results_template.format(qid=qid, docno=docid, sim=predicted, model=model))
            counter += 1
    if counter != len(id_list):
        print("Counter is not equal the total number of the documents")
        exit()
    results_file.flush()
    results_file.close()
    trec_eval_path = '/mnt/collections/p8shi/dev/biaffine/qa/trec_eval-9.0.5/trec_eval'
    trec_out = subprocess.check_output([trec_eval_path, gold_fname, pred_fname])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[5].split('\t')[-1])
    mean_reciprocal_rank = float(trec_out_lines[9].split('\t')[-1])
    p_30 = float(trec_out_lines[25].split('\t')[-1])

    #os.remove(pred_fname)

    return mean_average_precision, mean_reciprocal_rank, p_30

def Ranking_MAP_MRR(pairs, pred_fname="pred.txt", gold_fname=None, model="model"):
    if gold_fname == None:
        print("You need to pass filename of qrel to the function")
        exit()
    results_file = open(pred_fname, "w")
    results_template = "{qid} Q0 {docno} 0 {sim} {model}\n"
    counter = 0
    for pair in pairs:
        for qid, docno, sim in zip(pair[0], pair[1], pair[2]):
            results_file.write(results_template.format(qid=qid, docno=docno, sim=sim, model=model))
    results_file.flush()
    results_file.close()
    trec_eval_path = '/mnt/collections/p8shi/dev/biaffine/qa/trec_eval-9.0.5/trec_eval'
    trec_out = subprocess.check_output([trec_eval_path, gold_fname, pred_fname])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[5].split('\t')[-1])
    mean_reciprocal_rank = float(trec_out_lines[9].split('\t')[-1])
    p_30 = float(trec_out_lines[25].split('\t')[-1])
    return mean_average_precision, mean_reciprocal_rank, p_30




