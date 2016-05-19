#!/usr/bin/env python
import numpy as np

class SimpleFullWidthEncoder(object):

    def __init__(self, n_inputs=5**2, n_cats=4):
        self.n_inputs = n_inputs
        self.n_cats = n_cats

    def encode(self, i):
        offset = self.n_cats*i
        inputs = np.zeros(self.n_inputs)
        inputs[offset:offset+self.n_cats] = 1
        return inputs

