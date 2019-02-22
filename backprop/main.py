#!usr/bin/env python
# -*- coding: utf-8 -*-
# File: main.py

import numpy as np
from net import Net


def generate_data(n_feat, n_sample):
    n_class_1 = int(n_sample/2.)
    n_class_2 = n_sample - n_class_1
    x_1 = np.random.multivariate_normal(mean=1 * np.ones(n_feat), cov=0.02 * np.identity(n_feat), size=(n_class_1))
    x_2 = np.random.multivariate_normal(mean=-1 * np.ones(n_feat), cov=0.02 * np.identity(n_feat), size=(n_class_2))
    label_1 = np.ones(n_class_1)
    label_2 = np.zeros(n_class_2)

    x = np.concatenate((x_1, x_2), axis=0)
    y = np.concatenate((label_1, label_2), axis=0)

    idx = np.arange(n_sample)
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y

def one_hot(inputs, n_class):
    re_vec = np.zeros(shape=(len(inputs), n_class), dtype=np.int)
    re_vec[np.arange(len(inputs)), inputs.astype(np.int)] = 1
    return re_vec


if __name__ == '__main__':
    n_class = 2
    n_sample = 100
    train_x, train_y = generate_data(2, n_sample)
    # print(train_x)
    # print(train_y)

    net = Net(1, [10], input_len=2, n_class=2)
    net.create_net()
    out = net.forward(train_x)
    label = one_hot(train_y, n_class)
    cost = net.cost(label, out=None)
    print(cost)
