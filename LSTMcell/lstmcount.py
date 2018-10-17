#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lstmcount.py
# Author: Qian Ge <geqian1001@gmail.com>

# import random
# import math
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

SMALL_NUM = 1e-6

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def sigmoid(z):
    z = np.exp(-z)
    return 1 / (1 + z)

def identity(inputs):
    return inputs

def ceil_fnc(inputs):
    return np.ceil(inputs)

def one_hot(inputs, depth):
    n_inputs = len(inputs)
    one_hot_vector = np.zeros((n_inputs, depth))
    one_hot_vector[np.arange(n_inputs), inputs] = 1
    return one_hot_vector

class LSTMcell(object):
    def __init__(self, in_dim, out_dim, out_activation=identity):
        self._out_act = out_activation
        self.create_cell(in_dim, out_dim)

    def run_step(self, inputs, prev_state):
        g = tanh(np.matmul(inputs, self.wgx) + np.matmul(prev_state, self.wgh) + self.bg)
        i = sigmoid(np.matmul(inputs, self.wix) + np.matmul(prev_state, self.wih) + self.bi)
        f = sigmoid(np.matmul(inputs, self.wfx) + np.matmul(prev_state, self.wfh) + self.bf)
        o = sigmoid(np.matmul(inputs, self.wox) + np.matmul(prev_state, self.woh) + self.bo)
        state = np.multiply(g, i) + np.multiply(prev_state, f)
        print(i)
        # print('g: {:.2f}, i: {:.2f}, f: {:.2f}, o: {:.2f}, state: {:.2f}'
        #       .format(np.squeeze(g), np.squeeze(i), np.squeeze(f), np.squeeze(o), np.squeeze(state)))
        # print(np.matmul(inputs, self.wgx) + np.matmul(prev_state, self.wgh) + self.bg)
        return np.multiply(self._out_act(state), o)

    def create_cell(self, in_dim, out_dim):
        self.wgx = np.zeros((in_dim, out_dim))
        self.wgh = np.zeros((out_dim, out_dim))
        self.bg = np.zeros((1, out_dim))

        self.wix = np.zeros((in_dim, out_dim))
        self.wih = np.zeros((out_dim, out_dim))
        self.bi = np.zeros((1, out_dim))

        self.wfx = np.zeros((in_dim, out_dim))
        self.wfh = np.zeros((out_dim, out_dim))
        self.bf = np.zeros((1, out_dim))

        self.wox = np.zeros((in_dim, out_dim))
        self.woh = np.zeros((out_dim, out_dim))
        self.bo = np.zeros((1, out_dim))

    def set_config_by_name(self, name, val):
        """ Set config value by the dictionary name
        Args:
            name (string): key of dictionary
        """
        setattr(self, name, val)

    def display_param(self):
        print(self.wgx)

def assign_weight_count_all(cell, in_dim, out_dim):
    param_dict = {}
    param_dict['wgx'] = np.zeros((in_dim, out_dim))
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = 10. * np.ones((1, out_dim))

    param_dict['wix'] = [[10.] if i == 0 else [-10.] for i in range(10)]
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] =  np.zeros((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 10. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 10. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

def assign_weight_count_from_2(cell, in_dim, out_dim):
    param_dict = {}
    # param_dict['wgx'] = [[10.] if i == 0 or 2 else [-0.] for i in range(10)]
    input_count_num = [[10.] if i == 0 else [-0.] for i in range(10)]
    input_gate_num = [[10.] if i == 2 else [-0.] for i in range(10)]
    param_dict['wgx'] = np.concatenate((input_count_num, input_gate_num), axis=-1)
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    input_count_num = [[10.] if i == 0 else [-10.] for i in range(10)]
    input_gate_num = [[10.] if i == 2 else [-10.] for i in range(10)]
    param_dict['wix'] = np.concatenate((input_count_num, input_gate_num), axis=-1)
    param_dict['wih'] = [[100., 100.], [0., 0.]]
    param_dict['bi'] =  np.zeros((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 10. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 10. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

def assign_weight_count_from_2_stop_3(cell, in_dim, out_dim):
    param_dict = {}
    param_dict['wgx'] = np.zeros((in_dim, out_dim))
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = 10. * np.ones((1, out_dim))

    param_dict['wix'] = [[10.] if i == 0 else [-10.] for i in range(10)]
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] =  np.zeros((1, out_dim))

    param_dict['wfx'] = [[10.] if i == 2 else [-10.] for i in range(10)]
    param_dict['wfh'] = [[100.]]
    param_dict['bf'] = np.zeros((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 10. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

if __name__ == "__main__":
    cell = LSTMcell(in_dim=10, out_dim=2, out_activation=identity)
    assign_weight_count_from_2(cell, in_dim=10, out_dim=2)
    # cell.display_param()
    # x = [[1 if i == 2 else 0 for i in range(10)]]
    # count = cell.run_step(x, prev_state=[1.])

    # x = [[1 if i == 0 else 0 for i in range(10)]]
    # count = cell.run_step(x, prev_state=[1.])

    # test_list = [1,1,0,0,1,1,1,2,0,3,0,0,2,4,0,1,9,0,2,1]
    test_list = [1,1,0,0,2,0, 2]

    test_list = one_hot(test_list, depth=10)

    prev_state = [0., 0.]
    for d in test_list:
        # print(d)
        prev_state = cell.run_step([d], prev_state=prev_state)
        # print(prev_state)
        # print('state: {}'.format(np.squeeze(prev_state)))







