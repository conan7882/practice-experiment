#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import numpy as np


def fc_maxout(inputs, name='fc_maxout'):
    # inputs [batch_size, out_dim, k]
    with tf.name_scope(name):
        return tf.reduce_max(inputs, axis=-1)



# def fc(inputs, out_dim, nl=tf.identity, name='fc'):
#     inputs = batch_flatten(inputs)
#     in_dim = inputs.get_shape().as_list()[-1]
#     with tf.variable_scope(name):
#         w = tf.get_variable(
#             'weight', [in_dim, out_dim], trainable=True)
#         b = tf.get_variable(
#             'bias', [out_dim], trainable=True)
#         return nl(tf.matmul(inputs, w) + b)

# def softmax(inputs):
#     max_in = tf.reduce_max(inputs, axis=-1)
#     max_in = tf.tile(tf.reshape(max_in, (-1, 1)), [1, inputs.shape[-1]])
#     stable_in = inputs - max_in
#     normal_p = tf.reduce_sum(tf.exp(stable_in), axis=-1)
#     normal_p = tf.tile(tf.reshape(normal_p, (-1, 1)), [1, inputs.shape[-1]])
#     return tf.exp(stable_in) / normal_p

# def softplus(inputs):
#     return tf.log(1 + tf.exp(inputs))

# def batch_flatten(x):
#     """
#     Flatten the tensor except the first dimension.
#     """
#     shape = x.get_shape().as_list()[1:]
#     if None not in shape:
#         return tf.reshape(x, [-1, int(np.prod(shape))])
#     return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))



