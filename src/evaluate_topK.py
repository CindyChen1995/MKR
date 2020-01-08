'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_Ks = None
_sess = None


def evaluate_model(model, sess, testRatings, testNegatives, Ks, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _Ks
    global _sess
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _Ks = Ks
    _sess = sess
        
    # hits, ndcgs = [], []
    hits = {k: [] for k in _Ks}
    ndcgs = {k: [] for k in _Ks}

    # if num_thread > 1:  # Multi-thread
    #     pool = multiprocessing.Pool(processes=num_thread)
    #     res = pool.map(eval_one_rating, range(len(_testRatings)))
    #     pool.close()
    #     pool.join()
    #     hits = [r[0] for r in res]
    #     ndcgs = [r[1] for r in res]
    #     return (hits, ndcgs)
    # Single thread
    # for idx in range(len(_testRatings)):
    #     (hr, ndcg) = eval_one_rating(idx)
    #     hits.append(hr)
    #     ndcgs.append(ndcg)
    # return (hits, ndcgs)
    for idx in range(len(_testRatings)):
        eval_one_rating(idx, hits, ndcgs)
    return (hits, ndcgs)


def eval_one_rating(idx, hits, ndcgs):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')

    items, scores = _model.get_scores(_sess, {_model.user_indices: users,
                                              _model.item_indices: np.array(items),
                                              _model.head_indices: np.array(items)})

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = scores[i]
    # items.pop()

    # Evaluate top rank list
    # ranklist = heapq.nlargest(_Ks, map_item_score, key=map_item_score.get)
    # hr = getHitRatio(ranklist, gtItem)
    # ndcg = getNDCG(ranklist, gtItem)
    # return (hr, ndcg)

    # Evaluate top rank list
    ranklist = heapq.nlargest(_Ks[-1], map_item_score, key=map_item_score.get)
    for k in _Ks:
        hits[k].append(getHitRatio(ranklist[:k], gtItem))
        ndcgs[k].append(getNDCG(ranklist[:k], gtItem))


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
