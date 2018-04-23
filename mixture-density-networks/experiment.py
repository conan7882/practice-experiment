#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: experiment.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import dataflow
import mdn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Test dataset')

    return parser.parse_args()

def sample_mixture(p, mu, sigma, x):
    n_mix = p.shape[1]
    component_list = np.arange(n_mix)
    samples = []
    x_list = []
    for cur_p, cur_mu, cur_std, cur_x in zip(p, mu, sigma, x):
        for i in range(0, 2):
            pick_id = np.random.choice(component_list, p=cur_p)
            samples.append(
                np.random.normal(cur_mu[pick_id], cur_std[pick_id], 1))
            x_list.append(cur_x)
    return x_list, samples


if __name__ == '__main__':
    FLAGS = get_args()
    max_epoch = 300
    n_train = 3000
    # train_data = dataflow.MDData(n_train, 128)
    

    if FLAGS.data == 'moon':
        train_data = dataflow.Moon(n_train, 128)
        test_x = np.float32(np.arange(-1, 2, 0.01))
        
    elif FLAGS.data == 'gmm':
        train_data = dataflow.GMMData(centers=4,
                                      sigma=0.6,
                                      n_sample=n_train,
                                      batch_size=128)
        test_x = np.float32(np.arange(-3.5, 3.5, 0.01))
    else:
        train_data = dataflow.MDData(n_train, 128)
        test_x = np.float32(np.arange(-15.0, 15.0, 0.01))

    test_x = np.reshape(test_x, (len(test_x), 1))
    train_data.show_data()


    net = mdn.MDN(n_hidden=[10, 10], n_mixture=10, n_x_dim=1, n_y_dim=1)
    net.create_model()
    train_op = net.get_train_op()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        cur_epoch = 0
        loss_sum = 0
        step = 0
        while True:
            if cur_epoch < train_data.completed_epochs:
                print('epoch: {}, loss: {}'.format(cur_epoch, loss_sum / step))
                cur_epoch = train_data.completed_epochs
                step = 0
                loss_sum = 0
                if cur_epoch >= max_epoch:
                    break
            else:
                step += 1
            
            batch_data = train_data.next_batch()
            _, loss = sess.run([train_op, net.loss],
                         feed_dict={net.x: batch_data['x'],
                                    net.y: batch_data['y'],
                                    net.lr: 0.001})
            loss_sum += loss

        p, mu, sigma = sess.run([net.p, net.mu, net.sigma],
                                feed_dict={net.x: test_x})
        x, y = sample_mixture(p, mu, sigma, test_x)
        plt.plot(x, y, 'o')
    plt.show()
