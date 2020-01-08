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
from MLP import MLP
from evaluate import evaluate_model


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically


def train(args, train_data, test_data, test_negative, show_loss, log_dir, n_user, n_item):

    model = MLP(args, n_user, n_item)
    k_list = [1, 2, 5, 10, 20, 50, 100]

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
                _, loss = model.train_mlp(sess, get_feed_dict_for_mlp(model, train_data, start, start + args.batch_size, 0.5))

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
            hr = [np.mean(hits[k]) for k in k_list]
            ndcg = [np.mean(ndcgs[k]) for k in k_list]
            # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            # print('epoch: %d, HR = %.4f, NDCG = %.4f [%.1f]' % (step, hr, ndcg, int(time.time())))
            topk_str = 'epoch: %d' % step
            topk_str += '\thr: '
            for i in hr:
                print('%.4f\t' % i, end='')
                topk_str += '%.4f\t' % i
            topk_str += '\tndcg: '
            for i in ndcg:
                print('%.4f\t' % i, end='')
                topk_str += '%.4f\t' % i
            print(topk_str)


def get_feed_dict_for_mlp(model, data, start, end, keep_drop=0.0):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.keep_drop: keep_drop}
    return feed_dict
