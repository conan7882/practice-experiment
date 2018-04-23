#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mdn.py
# Author: Qian Ge <geqian1001@gmail.com>

import math
import tensorflow as tf
import layers


class MDN(object):
    def __init__(self, n_hidden, n_mixture, n_x_dim=1, n_y_dim=1):
        assert n_y_dim == 1
        if not isinstance(n_hidden, list):
            n_hidden = [n_hidden]
        self._n_hidden = n_hidden
        self._n_mix = n_mixture
        self._x_dim = n_x_dim
        self._y_dim = n_y_dim

    def create_model(self):
        self._create_input()
        self._create_model()

    def _create_input(self):
        self.x = tf.placeholder(tf.float32, [None, self._x_dim], name='x')
        self.y = tf.placeholder(tf.float32, [None, self._y_dim], name='y')

    def _create_model(self):
        inputs = self.x

        for idx, n_hidden in enumerate(self._n_hidden):
            outputs = layers.fc(
                inputs, n_hidden, nl=tf.tanh, name='hidden_{}'.format(idx))
            inputs = outputs

        n_out_dim = self._n_mix * (1 + self._y_dim + self._y_dim)
        outputs = layers.fc(
                inputs, n_out_dim, name='out_layer')

        self.p, self.mu, self.sigma = self.get_mix_model_params(outputs)

    def get_mix_model_params(self, inputs):
        p_in = inputs[:, :self._n_mix]
        mu_in = inputs[:, self._n_mix: self._n_mix * (1 + self._y_dim)]
        sigma_in = inputs[:, self._n_mix * (1 + self._y_dim):]

        # p_out = tf.nn.softmax(p_in, dim=-1)
        p_out = layers.softmax(p_in)
        mu_out = mu_in
        sigma_out = layers.softplus(sigma_in)

        return p_out, mu_out, sigma_out

    def get_gaussian_prob(self, inputs):
        result = inputs - self.mu
        result = tf.multiply(result, tf.reciprocal(self.sigma))
        result = -tf.square(result) / 2
        result = tf.multiply(tf.exp(result), tf.reciprocal(self.sigma))\
                      * (1 / math.sqrt(2 * math.pi))
        return result

    def get_mix_model(self):
        #TODO need to write for multidim
        prob = self.get_gaussian_prob(self.y)
        return tf.reduce_sum(
            tf.multiply(prob, self.p), 1, keep_dims=True)

    def get_loss(self):
        result = self.get_mix_model()
        loss = -tf.log(result)
        return tf.reduce_mean(loss)

    def get_train_op(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.loss = self.get_loss()
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        return opt.minimize(self.loss)
