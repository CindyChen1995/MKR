# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   Author :       cmy
   date：          2020/1/2
-------------------------------------------------
"""
import argparse
import numpy as np
from load_data import get_data
from train_dmf import train
import pickle
from preprocess1 import Dataset, getTrainMatrix, get_train_instances


def parse_args():
    parser = argparse.ArgumentParser(description="Run DMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movie',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--user_layers', nargs='?', default='[512, 64]',
                        help="Size of each user layer")
    parser.add_argument('--item_layers', nargs='?', default='[1024, 64]',
                        help="Size of each item layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--dim', type=int, default=16,
                        help='Whether to save the trained model.')
    parser.add_argument('--l2_weight', type=float, default=1e-6,
                        help='weight of l2 regularization')
    return parser.parse_args()


if __name__ == "__main__":
    show_loss = True
    show_topk = True
    log_dir = '../dmf_logs/'

    args = parse_args()
    # data = get_data(args)
    # train(args, data, show_loss, show_topk, log_dir)

    dataset = Dataset('data/ml-1m')
    train_data, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train_data.shape
    # train_matrix = getTrainMatrix(train_data)
    try:
        users, items, labels = pickle.load(open('data/data_negative.p', mode='rb'))
    except:
        users, items, labels = get_train_instances(train_data, args.num_neg, num_items)
        pickle.dump((users, items, labels), open('data/data_negative.p', mode='wb'))
    users = np.reshape(users, [-1, 1])
    items = np.reshape(items, [-1, 1])
    labels = np.reshape(labels, [-1, 1])
    data = np.concatenate([users, items, labels], axis=1)
    train(args, data, testRatings, testNegatives, show_loss, show_topk, log_dir, num_users, num_items)
