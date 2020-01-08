# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   Author :       cmy
   date：          2020/1/4
-------------------------------------------------
"""
import argparse
from train_dmf_mlp import train
from data_loader2 import load_rating


def parse_args():
    parser = argparse.ArgumentParser(description="Run DMF+MLP.")
    parser.add_argument('--path', nargs='?', default='../../data/',
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
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--dmf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for DMF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


if __name__ == "__main__":
    show_loss = True
    show_topk = True
    log_dir = 'dmf_mlp_logs/'

    args = parse_args()

    data = load_rating(args)
    train(args, data, show_loss, log_dir)