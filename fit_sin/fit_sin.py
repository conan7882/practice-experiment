#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fit_sin.py
# Author: Qian Ge <geqian1001@gmail.com>

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def generate_data(n_sample=500):
    x_list = [i + 0.05 * np.random.randn() for i in np.linspace(-15 * np.pi, 15 * np.pi, num=n_sample)]
    y_list = [np.sin(x) + 0.1 * np.random.randn() for x in x_list]

    x_list = np.expand_dims(x_list, axis=-1)

    # plt.figure()
    # plt.plot(x_list, y_list, '.')
    # plt.show()

    return x_list, y_list

def generate_approx_sine(n_degree, n_sample=10):
    x_list = [i for i in np.linspace(-10 * np.pi, 10 * np.pi, num=n_sample)]
    y_list = [np.sin(x) for x in x_list]
    
    y_list_ = np.array([0. for i in range(n_sample)])
    for n in range(n_degree):
        y_list_ += np.array([((-1) ** n) / (math.factorial(2. * n + 1)) * x ** (2. * n + 1) for x in x_list])
    return x_list, y_list, y_list_

def suffle_data(x, y):
    data_idx = [i for i in range(np.array(x).shape[0])]
    # print(data_idx)
    random.shuffle(data_idx)
    x = [x[i] for i in data_idx]
    y = [y[i] for i in data_idx]
    return x, y

def identity(inputs):
    return inputs

def linear(layer_dict, out_dim, name, wd=0, init_w=tf.keras.initializers.he_normal(), init_b=None, nl=identity):
    inputs = layer_dict['cur_input']
    in_dim = inputs.get_shape().as_list()[1]

    if wd > 0:
        regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
    else:
        regularizer=None

    with tf.variable_scope(name):
        weights = tf.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  initializer=init_w,
                                  regularizer=regularizer,
                                  trainable=True)
        biases = tf.get_variable('biases',
                                 shape=[out_dim],
                                 initializer=init_b,
                                 regularizer=None,
                                 trainable=True)
    act = tf.nn.xw_plus_b(inputs, weights, biases)
    result = nl(act, name='output')
    layer_dict['cur_input'] = result
    print(weights)
    return result

def model(x):
    layer_dict = {}
    layer_dict['cur_input'] = x
    for i in range(5):
        linear(layer_dict, out_dim=10, nl=tf.nn.sigmoid, name='fc{}'.format(i))
    linear(layer_dict, out_dim=1, nl=tf.tanh, name='out')
    y_hat = tf.squeeze(layer_dict['cur_input'], axis=-1)
    return y_hat

def train_op(y_hat, y_label, lr):
    l2_loss = tf.reduce_mean((y_hat - y_label) ** 2)
    opt = tf.train.AdamOptimizer(lr)
    # opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
    var_list = tf.trainable_variables()
    grads = tf.gradients(l2_loss, var_list)
    train_op = opt.apply_gradients(zip(grads, var_list))
    return train_op, l2_loss

if __name__ == '__main__':
    train_x = []
    train_y = []
    

    for i in range(0, 2):
        x_list, y_list = generate_data()
        train_x.extend(x_list)
        train_y.extend(y_list)

    batch_size = 128
    n_training = np.array(train_y).shape[0]
    n_step_per_epoch = np.floor(n_training / batch_size).astype(int)

    x_pl = tf.placeholder(tf.float32, [None, 1], name='x')
    y_pl = tf.placeholder(tf.float32, [None], name='y')
    lr = tf.placeholder(tf.float32, name='lr')

    y_hat = model(x_pl)
    train_op, l2_loss = train_op(y_hat, y_pl, lr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1, 5000):
            train_x, train_y = suffle_data(train_x, train_y)
            start_id = 0
            loss_sum = 0
            for step in range(n_step_per_epoch):
                cur_x = train_x[start_id: batch_size]
                cur_y = train_y[start_id: batch_size]
                start_id += batch_size
                _, y_, loss = sess.run([train_op, y_hat, l2_loss], feed_dict={x_pl: train_x, y_pl: train_y, lr: 1e-2})
                loss_sum += loss
            print(i, loss_sum / n_step_per_epoch)
        # for i in range(1, 5000):
        #     # train_x, train_y = suffle_data(train_x, train_y)
        #     _, y_, loss = sess.run([train_op, y_hat, l2_loss], feed_dict={x_pl: train_x, y_pl: train_y, lr: 1e-3})
        #     print(loss)
        # for i in range(1, 5000):
        #     # train_x, train_y = suffle_data(train_x, train_y)
        #     _, y_, loss = sess.run([train_op, y_hat, l2_loss], feed_dict={x_pl: train_x, y_pl: train_y, lr: 1e-3})
        #     print(loss)

        x_ = [i for i in np.linspace(-15 * np.pi, 15 * np.pi, num=300)]
        x_ = np.expand_dims(x_, axis=-1)
        y_ = sess.run(y_hat, feed_dict={x_pl: x_})
        plt.figure()
        plt.plot(np.squeeze(x_), np.squeeze(y_), '.')
        plt.plot(np.squeeze(train_x), train_y, '.')
        # plt.show()

        train_x, train_y, train_y_ = generate_approx_sine(n_degree=5, n_sample=500)
        plt.figure()
        plt.plot(train_x, train_y, '.')
        plt.plot(train_x, train_y_, '.')
        plt.ylim([-1.1, 1.1])
        plt.show()





