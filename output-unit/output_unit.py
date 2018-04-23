#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: output_unit.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.reshape(x, (-1))
    return np.exp(x) / (1 + np.exp(x))

def d_logsigmoid(x):
    return -sigmoid(-x)

def softmax(x):
    stabel_x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def d_logsoftmax(x, true_idx):
    d_list = np.exp(x) / np.sum(np.exp(x))
    d_list[true_idx] = np.exp(x[true_idx]) / np.sum(np.exp(x)) - 1
    return d_list


if __name__ == '__main__':
    x = np.float32(np.arange(-5.0, 5.0, 0.01))
    sigmoid_x = sigmoid(x)

    plt.figure()
    plt.plot(x, sigmoid(x), label='sigmoid')
    plt.plot(x, -np.log(sigmoid_x), label='loss function')
    plt.plot(x, d_logsigmoid(x), label='gradient')
    plt.title('correct label y = 1')
    plt.xlabel('x')
    plt.legend()

    plt.figure()
    plt.plot(x, 1 - sigmoid(x), label='sigmoid')
    plt.plot(x, -np.log(1 - sigmoid_x), label='loss function')
    plt.plot(x, d_logsigmoid(-x), label='gradient')
    plt.title('correct label y = 0')
    plt.xlabel('x')
    plt.legend()

    x = -5.0
    cnt = 0
    x_list = []
    d_list = []
    while True:
        x_list.append(x)
        cnt += 1
        if cnt == 6000:
            break
        d = d_logsigmoid(x)
        d_list.append(d)
        x = -0.1 * d + x

    plt.figure()
    plt.plot(x_list)
    plt.xlabel('step')
    plt.title('Find optimal x to minimize loss function when correct label y = 1')

    x = [1, 2, 3]
    cnt = 0
    x_list = []
    d_list = []
    while True:
        x_list.append(x)
        cnt += 1
        if cnt == 500:
            break
        d = d_logsoftmax(x, 0)
        d_list.append(d)
        x = -0.1 * d + x

    x_list = np.array(x_list)
    d_list = np.array(d_list)

    plt.figure()
    plt.plot(x_list[:, 0], label='true class')
    plt.plot(x_list[:, 1], label='wrong class')
    plt.plot(x_list[:, 2], label='wrong class')
    
    plt.legend()
    plt.title('Find optimal x to minimize loss function')

    plt.figure()
    plt.plot(d_list[:, 0], label='true class')
    plt.plot(d_list[:, 1], label='wrong class')
    plt.plot(d_list[:, 2], label='wrong class')
    plt.title('gradient')
    plt.legend()

    plt.show()
