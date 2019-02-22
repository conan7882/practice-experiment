#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: net.py
# Author: Qian Ge

import numpy as np


class Net(object):
    def __init__(self, n_hidden, hidden_unit_list, input_len, n_class):
        if not isinstance(hidden_unit_list, list):
            hidden_unit_list = [hidden_unit_list]
        assert n_hidden == len(hidden_unit_list)
        self._n_hidden = n_hidden
        # self._hidden_list = hidden_unit_list
        self._n_class = n_class
        self._input_len = input_len

        self._unit_list = [input_len] + hidden_unit_list + [n_class]

    def create_net(self):
        self.weight = []
        self.bias = []
        unit_list = []
        for layer_id, n_in in enumerate(self._unit_list[:-1]):
            n_out = self._unit_list[layer_id + 1]
            weight = np.random.normal(loc=0., scale=0.002, size=(n_in, n_out))
            bias = np.zeros((n_out,))
            self.weight.append(weight)
            self.bias.append(bias)

        print('Model Summary:')
        for layer_id, (w, b) in enumerate(zip(self.weight, self.bias)):
            print('Layer_{}:\t W: ({}, {}) \t B: ({})'.format(layer_id, w.shape[0], w.shape[1], b.shape[0]))


    def act_fnc(self, inputs):
        return 1. / (1. + np.exp(-inputs))
        # if inputs >= 0:
        #     return 1. / (1. + np.exp(-inputs))
        # else:
        #     return np.exp(inputs) / (1. + np.exp(inputs))

    def forward(self, inputs):
        self.activation = []
        prev_in = inputs
        for layer_id, (w, b) in enumerate(zip(self.weight, self.bias)):
            act_out = self.act_fnc(np.matmul(prev_in, w) + b)
            prev_in = act_out
            self.activation.append(act_out) 
            
    def cost(self, label, out=None):
        if out is None:
            out = self.activation[-1] # [bsize, n_class]
        cost = -np.multiply(label, np.log(out)) - np.multiply((1 - label), np.log(1 - out))
        return np.mean(cost)

    @property
    def out(self):
        return self.activation[-1]


