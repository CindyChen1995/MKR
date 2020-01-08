# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   Author :       cmy
   date：          2020/1/2
-------------------------------------------------
"""
import datetime
import heapq
import numpy as np
import tensorflow as tf
import time

from metrics import ndcg_at_k
from train import get_user_record
from DMF import DMF
from evaluate import evaluate_model


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically


# def train(args, data, show_loss, show_topk, log_dir):
#     n_user, n_item = data[0], data[1]
#     train_data, eval_data, test_data = data[2], data[3], data[4]
def train(args, train_data, test_data, test_negative, show_loss, show_topk, log_dir, n_user, n_item):

    model = DMF(args, n_user, n_item)
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    # train_record = get_user_record(train_data, True)
    # test_record = get_user_record(test_data, False)
    # user_list = list(set(train_record.keys()) & set(test_record.keys()))
    # if len(user_list) > user_num:
    #     user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))

    # with tf.Session(config=config) as sess,\
    #     open(log_dir + 'result_' + str(args.epochs) + '_' + str(args.lr) + '_' + str(int(time.time())) + '.txt', 'w') as f_result:
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.epochs):
            # f_result.write('**************************epoch_i:' + str(step) + '********************' + '\n')
            # RS training
            np.random.shuffle(train_data)
            start = 0
            batch_i = 0
            while start < train_data.shape[0]:
                _, loss = model.train_dmf(sess, get_feed_dict_for_dmf(model, train_data, start, start + args.batch_size, 0.5))

                start += args.batch_size
                if show_loss:
                    if (step * (len(train_data) // args.batch_size) + batch_i) % 20 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            time_str,
                            step,
                            batch_i,
                            (len(train_data) // args.batch_size),
                            loss))
                    # print(loss)
                batch_i += 1

            (hits, ndcgs) = evaluate_model(model, sess, test_data, test_negative, 10, 1)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('epoch: %d, HR = %.4f, NDCG = %.4f [%.1f]' % (step, hr, ndcg, int(time.time())))

            # CTR evaluation
            # train_auc, train_acc = model.eval(sess, get_feed_dict_for_dmf(model, train_data, 0, train_data.shape[0]))
            # eval_auc, eval_acc = model.eval(sess, get_feed_dict_for_dmf(model, eval_data, 0, eval_data.shape[0]))
            # test_auc, test_acc = model.eval(sess, get_feed_dict_for_dmf(model, test_data, 0, test_data.shape[0]))

            # eval_str = 'epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f' \
            #            % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc)
            # eval_str = 'epoch %d    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f' \
            #            % (step, eval_auc, eval_acc, test_auc, test_acc)
            # print(eval_str)
            # f_result.write(eval_str + '\n')

            # top-K evaluation
            # if show_topk:
            #     topk_str = ''
            #     precision, recall, f1, hr, ndcg = topk_eval(
            #         sess, model, user_list, train_record, test_record, item_set, k_list)
            #     print('precision: ', end='')
            #     topk_str += 'precision: '
            #     for i in precision:
            #         print('%.4f\t' % i, end='')
            #         topk_str += '%.4f\t' % i
            #     print()
            #     print('recall: ', end='')
            #     topk_str += '\n' + 'recall: '
            #     for i in recall:
            #         print('%.4f\t' % i, end='')
            #         topk_str += '%.4f\t' % i
            #     print()
            #     print('f1: ', end='')
            #     topk_str += '\n' + 'f1: '
            #     for i in f1:
            #         print('%.4f\t' % i, end='')
            #         topk_str += '%.4f\t' % i
            #     print()
            #     print('hr: ', end='')
            #     topk_str += '\n' + 'hr: '
            #     for i in hr:
            #         print('%.4f\t' % i, end='')
            #         topk_str += '%.4f\t' % i
            #     print()
            #     print('ndcg: ', end='')
            #     topk_str += '\n' + 'ndcg: '
            #     for i in ndcg:
            #         print('%.4f\t' % i, end='')
            #         topk_str += '%.4f\t' % i
            #     print()
            #     f_result.write(topk_str + '\n')


def get_feed_dict_for_dmf(model, data, start, end, keep_drop=0.0):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.keep_drop: keep_drop}
    return feed_dict


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    hr_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}
    total_test = 0

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        items, scores = model.get_scores(sess, {model.user_indices: [user] * len(test_item_list),
                                                model.item_indices: test_item_list, model.keep_drop: 0.0})
        for item, score in zip(items, scores):
            item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        K_max_item_score = heapq.nlargest(k_list[-1], item_score_map, key=item_score_map.get)
        r = []
        for i in K_max_item_score:
            if i in test_record[user]:
                r.append(1)
            else:
                r.append(0)

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

            hr_list[k].append(hit_num)
            ndcg_list[k].append(ndcg_at_k(r, k))

        total_test += len(test_record[user])

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]
    hr = [np.sum(hr_list[k]) / total_test for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, f1, hr, ndcg