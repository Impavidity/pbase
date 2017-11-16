def get_span(label):
    span = []
    st = -1
    en = -1
    flag = False
    for k, item in enumerate(label):
        if item == 'I' and flag == False:
            flag = True
            st = k
        if item != 'I' and flag == True:
            flag = False
            en = k
            span.append((st, en))
            st = -1
            en = -1
    if st != -1 and en == -1:
        en = len(label)
        span.append((st, en))
    return span


def NEREvaluation(goldLabel, predLabel):
    if len(goldLabel) != len(predLabel):
        print("Length is not matched {}/{}".format(len(goldLabel), len(predLabel)))
    right = 0
    pred_en, total_en = 0, 0
    for gold, pred in zip(goldLabel, predLabel):
        gold_span = get_span(gold)
        pred_span = get_span(pred)
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




