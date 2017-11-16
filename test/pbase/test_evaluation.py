import unittest
from pbase import evaluation

class TestScript(unittest.TestCase):
    def test_NEREvaluation(self):
        fin = open("/u1/p8shi/pytorch/CleanBubo/entity_detection/crf/stanford-ner/data/stanford.predicted.valid")
        gold_list = []
        pred_list = []
        gold = []
        pred = []
        sent_num = 0
        word_num = 0
        for line in fin.readlines():
            if len(line) == 1:
                gold_list.append(gold)
                pred_list.append(pred)
                sent_num += 1
                word_num += len(gold)
                gold = []
                pred = []
            else:
                word, gold_label, pred_label = line.strip().split()
                gold.append(gold_label)
                pred.append(pred_label)
        if gold != [] or pred != []:
            gold_list.append(gold)
            pred_list.append(pred)
        p, r, f1 = evaluation.NEREvaluation(gold_list, pred_list)
        print(p, r, f1)



if __name__=="__main__":
    unittest.main()

