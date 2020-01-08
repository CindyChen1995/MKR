# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   Author :       cmy
   date：          2020/1/4
-------------------------------------------------
"""
import argparse
import datetime
import logging
import numpy as np
import tensorflow as tf
from time import time
from process_data import load_rating, get_train_instances, getTrainMatrix
from DeepCF.evaluate import evaluate_model
from DeepCF.model import Model


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically


def parse_args():
    parser = argparse.ArgumentParser(description="Run DMF+MLP.")
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movie',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[512,256,128,64]',
                        help="MLP layers. Note that the first layer is the concatenation "
                             "of user and item embeddings. So layers[0]//2 is the embedding size.")
    parser.add_argument('--user_layers', nargs='?', default='[512, 64]',
                        help="Size of each user layer")
    parser.add_argument('--item_layers', nargs='?', default='[1024, 64]',
                        help="Size of each item layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    # parser.add_argument('--lr_dmf', type=float, default=0.0001,
    #                     help='DMF Learning rate.')
    parser.add_argument('--drop_out', type=float, default=0.5,
                        help='drop_out rate.')
    parser.add_argument('--l2_weight', type=float, default=1e-6,
                        help='weight of l2 regularization')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    parser.add_argument('--dmf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for DMF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


def get_feed_dict(model, data, start, end, drop_out=0.0):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.drop_out: drop_out}
    return feed_dict


if __name__ == "__main__":
    show_loss = True
    show_topk = True
    args = parse_args()

    logging.basicConfig(filename='logs/%s_DMF_MLP_%d.log' % (args.dataset, time()),
                        level=logging.INFO, filemode='w')

    n_user, n_item, train_data, test_data, test_negative = load_rating(args)
    train_matrix = getTrainMatrix(train_data)
    model = Model(args, n_user, n_item, train_matrix, train_matrix.T)
    k = 10
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        logging.info('DMF+MLP arguments: %s\n' % args)
        for step in range(args.epochs):
            # RS training
            data = get_train_instances(train_data, args.num_neg, n_item)

            t1 = time()
            start = 0
            batch_i = 0
            while start < data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, data, start, start + args.batch_size, 0.5))
                start += args.batch_size
                if show_loss:
                    if (step * (len(data) // args.batch_size) + batch_i) % 50 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            time_str,
                            step,
                            batch_i,
                            (len(data) // args.batch_size),
                            loss))
                    # print(loss)
                batch_i += 1

            t2 = time()
            (hits, ndcgs) = evaluate_model(model, sess, test_data, test_negative, k, 1)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            str1 = 'Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' \
                   % (step, t2 - t1, hr, ndcg, loss, time() - t2)
            print(str1)
            logging.info(str1)