#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataflow.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_moons


def make_list(data):
    if not isinstance(data, list):
        data = [data]
    return data


class MDData(object):
    # generate data the same as 
    # http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
    def __init__(self, n_sample, batch_size):
        self._bsize = batch_size
        self._n_sample = n_sample

        self.y = np.float32(np.random.uniform(-10.5, 10.5, (1, n_sample))).T
        r_data = np.float32(np.random.normal(size=(n_sample,1)))
        self.x = np.float32(np.sin(0.75 * self.y) * 7.0
                            + self.y * 0.5 + r_data * 1.0)

        self._data_id = 0
        self._data_idx = np.arange(n_sample)
        self.completed_epochs = 0

    def show_data(self):
        import matplotlib.pyplot as plt
        plt.plot(self.x, self.y, 'o')
        # plt.show()

    def next_batch(self):
        assert self._bsize <= self._n_sample

        if self._data_id + self._bsize >= self._n_sample:
            self._data_id = 0
            self.completed_epochs += 1
            np.random.shuffle(self._data_idx)

        start = self._data_id
        end = self._data_id + self._bsize
        self._data_id = end

        return {'x': self.x[self._data_idx[start: end]],
                'y': self.y[self._data_idx[start: end]]}


class GMMData(MDData):
    def __init__(self, centers, sigma, n_sample, batch_size):
        self._bsize = batch_size
        self._n_sample = n_sample

        sampels, _ = make_blobs(
            n_samples=n_sample, centers=centers,
            cluster_std=sigma, random_state=0)
        self.x = np.reshape(sampels[:, 0], (n_sample, 1))
        self.y = np.reshape(sampels[:, 1], (n_sample, 1))

        self._data_id = 0
        self._data_idx = np.arange(n_sample)
        self.completed_epochs = 0


class Moon(MDData):
    def __init__(self, n_sample, batch_size):
        self._bsize = batch_size
        self._n_sample = n_sample

        sampels, _ = make_moons(n_sample, noise=.05, random_state=0)
        self.x = np.reshape(sampels[:, 0], (n_sample, 1))
        self.y = np.reshape(sampels[:, 1], (n_sample, 1))

        self._data_id = 0
        self._data_idx = np.arange(n_sample)
        self.completed_epochs = 0



