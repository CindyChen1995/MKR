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
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from layers import Dense

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically


class Model(object):
    def __init__(self, args, n_users, n_items, user_matrix, item_matrix):
        self._parse_args(args, n_users, n_items, user_matrix, item_matrix)
        self._build_inputs()
        self._build_model()
        self._build_loss(args)
        self._build_train(args)

    def _parse_args(self, args, n_users, n_items, user_matrix, item_matrix):
        self.n_user = n_users
        self.n_item = n_items
        self.user_matrix = tf.constant(user_matrix)
        self.item_matrix = tf.constant(item_matrix)

        self.user_layers = eval(args.user_layers)
        self.item_layers = eval(args.item_layers)
        self.layers = eval(args.layers)

        # for computing l2 loss
        self.vars_mlp = []
        self.vars_dmf = []

    def _build_inputs(self):
        self.user_indices = tf.placeholder(tf.int32, [None], 'user_indices')
        self.item_indices = tf.placeholder(tf.int32, [None], 'item_indices')
        self.labels = tf.placeholder(tf.float32, [None], 'labels')
        self.drop_out = tf.placeholder(tf.float32)

    def _build_model(self):
        self.user_input = tf.nn.embedding_lookup(self.user_matrix, self.user_indices)
        self.item_input = tf.nn.embedding_lookup(self.item_matrix, self.item_indices)

        self._build_mlp_layers()
        self._build_dmf_layers()
        self._build_predict()

    def _build_mlp_layers(self):
        user_layer = Dense(self.n_item, self.layers[0] // 2, act=None, name='user_embedding')
        item_layer = Dense(self.n_user, self.layers[0] // 2, act=None, name='item_embedding')
        self.mlp_user_embeddings = user_layer(self.user_input)
        self.mlp_item_embeddings = item_layer(self.item_input)
        self.mlp_vector = tf.concat([self.mlp_user_embeddings, self.mlp_item_embeddings], axis=1)

        self.vars_mlp.extend(user_layer.vars)
        self.vars_mlp.extend(item_layer.vars)

        for i in range(1, len(self.layers)):
            mlp_layer = Dense(input_dim=self.layers[i-1], output_dim=self.layers[i], dropout=self.drop_out,
                              name='layer%d' % i)
            self.mlp_vector = mlp_layer(self.mlp_vector)

            self.vars_mlp.extend(mlp_layer.vars)

    def _build_dmf_layers(self):
        user_layer = Dense(self.n_item, self.user_layers[0], act=None, name='user_layer0')
        item_layer = Dense(self.n_user, self.item_layers[0], act=None, name='item_layer0')
        self.user_vector = user_layer(self.user_input)
        self.item_vector = item_layer(self.item_input)

        for i in range(1, len(self.user_layers)):
            self.vars_dmf.extend(user_layer.vars)

            user_layer = Dense(input_dim=self.user_layers[i-1], output_dim=self.user_layers[i], name='user_layer%d' % i)
            self.user_vector = user_layer(self.user_vector)

        for i in range(1, len(self.item_layers)):
            self.vars_dmf.extend(item_layer.vars)
            item_layer = Dense(input_dim=self.item_layers[i-1], output_dim=self.item_layers[i], name='item_layer%d' % i)
            self.item_vector = item_layer(self.item_vector)

        self.dmf_vector = tf.multiply(self.user_vector, self.item_vector)

    def _build_predict(self):
        self.predict_vector = tf.concat([self.dmf_vector, self.mlp_vector], axis=1)
        predict_layer = tf.layers.Dense(1, name='predict_layer')

        self.predict_vector = predict_layer(self.predict_vector)

        self.predict_vector = tf.reshape(self.predict_vector, [-1])

        self.predict_normalized = tf.nn.sigmoid(self.predict_vector)

    def _build_loss(self, args):
        self.base_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.predict_vector))
        # self.l2_loss = tf.nn.l2_loss(self.dmf_user_embeddings) + tf.nn.l2_loss(self.dmf_user_embeddings) +\
        #                tf.nn.l2_loss(self.mlp_user_embeddings) + tf.nn.l2_loss(self.mlp_item_embeddings)
        self.l2_loss = None
        for var in self.vars_dmf:
            if self.l2_loss is None:
                self.l2_loss = tf.nn.l2_loss(var)
                continue
            self.l2_loss += tf.nn.l2_loss(var)
        for var in self.vars_mlp:
            self.l2_loss += tf.nn.l2_loss(var)
        self.loss = self.base_loss + self.l2_loss * args.l2_weight

    def _build_train(self, args):
        self.optimizer = tf.train.AdamOptimizer(args.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.predict_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.predict_normalized], feed_dict)

    def load_pretrain_dmf(self, model, dmf_model, dmf_layers):
        dmf_user_embeddings = dmf_model.get_layer('dmf_user_embedding').get_weights()
        dmf_item_embeddings = dmf_model.get_layer('dmf_item_embedding').get_wieghts()

        model.get_layer('dmf_user_embedding').set_weights(dmf_user_embeddings)
        model.get_layer('dmf_item_embedding').set_weights(dmf_item_embeddings)

        for i in range(1, len(dmf_layers)):
            dmf_user_layer_weights = dmf_model.get_layer('user_layer%d' % i).get_weights()
            model.get_layer('user_layer%d' % i).set_weights(dmf_user_layer_weights)

            dmf_item_layer_weights = dmf_model.get_layer('item_layer%d' % i).get_weights()
            model.get_layer('item_layer%d' % i).set_weights(dmf_item_layer_weights)

        dmf_prediction = dmf_model.get_layer('predict_layer').get_weights()
        new_weights = np.concatenate((dmf_prediction[0], np.array([[0, ]] * dmf_layers[-1])), axis=0)
        new_b = dmf_prediction[1]
        model.get_layer('predict_layer').set_weights([new_weights, new_b])

        return model

    def load_pretrain_mlp(self, model, mlp_model, mlp_layers):
        mlp_user_embeddings = mlp_model.get_layer('mlp_user_embedding').get_weights()
        mlp_item_embeddings = mlp_model.get_layer('mlp_item_embedding').get_wieghts()

        model.get_layer('mlp_user_embedding').set_weights(mlp_user_embeddings)
        model.get_layer('mlp_item_embedding').set_weights(mlp_item_embeddings)

        for i in range(1, len(mlp_model)):
            mlp_layer_weights = mlp_model.get_layer('layer%d' % i)
            model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

        dmf_prediction = model.get_layer('predict_layer').get_weights()
        mlp_prediction = mlp_model.get_layer('predict_layer').get_weights()

        new_weights = np.concatenate((dmf_prediction[0][: mlp_layers[-1]], mlp_prediction[0]), axis=0)
        new_b = dmf_prediction[1] + mlp_prediction[1]
        # 0.5 means the contributions of MF and MLP are equal
        model.get_layer('predict_layer').set_weights([0.5 * new_weights, 0.5 * new_b])
        return model
