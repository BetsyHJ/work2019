import numpy as np
import math

def eval_one_rating(predictions):
    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict > pos_predict).sum()

    K = [1, 3, 5, 10, 15, 20]
    hr, ndcg = {}, {}
    for k in K:
        hr[k] = _getHR(position, k)
        ndcg[k] = _getNDCG(position, k)
    mrr = 1.0 / (position+1)
    return (hr, ndcg, mrr)

def _getHR(location, K):
    if location < K:
        return 1.0
    return 0.0
def _getNDCG(location, K):
    if location < K:
        return math.log(2) / math.log(location+2)
    return 0

def eval_multi_rating(scores, gtItem, cans):
    K = [1, 3, 5, 10, 15, 20]
    ranklist = {}
    for i in range(len(cans)):
        ranklist[cans[i]] = scores[i]
    midt = sorted(ranklist.items(), key=lambda d: d[1], reverse=True)
    ranklist = []
    for i, _ in midt:
        ranklist.append(i)
    #print 'location', location
    P, R, NDCG = {}, {}, {}
    for k in K:
        right = getP(ranklist[:k], gtItem) 
        P[k] = right * 1.0 / k
        R[k] = right * 1.0 / len(gtItem)
        NDCG[k] = getNDCG(ranklist[:k], gtItem)
    mrr = getMrr(ranklist, gtItem)
    return P, R, NDCG, mrr

def getMrr(ranklist, gtItem):
    for c in range(len(ranklist)):
        if ranklist[c] in gtItem:
            return 1.0 / (c + 1.0)
    return 0.0

def getP(ranklist, gtItem):
    P = 0.0
    for i in ranklist:
        if i in gtItem:
            P += 1
    return P

def getNDCG(ranklist, trueResult):
    ranklabel = []
    for i in ranklist:
        if i in trueResult:
            ranklabel.append(1)
        else:
            ranklabel.append(0)
    DCG=0.0
    for k,i in enumerate(ranklist):
        DCG+=(pow(2,ranklabel[k])-1)/np.log2(1+k+1)
    IDCG=0.0
    rank_pair=zip(ranklist,ranklabel)
    rank_pair=sorted(rank_pair,key=lambda s:s[1],reverse=True)
    for k,i in enumerate(rank_pair):
        IDCG+=(pow(2,i[1])-1)/np.log2(1+k+1)
    if IDCG==0.0:
        return 0
    else:
        return DCG/IDCG