#!/usr/bin/env python

import numpy as np

class Brain():

    def __init__(self, input_dim=5, output_dim=2, n_hidden=2, hidden_dim=10):
    	'''
    	Currently we aren't holding state in neurons
    	'''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.layers = []
        for hl in range(self.n_hidden):
            d_in = self.input_dim if hl == 0 else self.hidden_dim
            d_out = self.hidden_dim
            l = Layer(d_in, d_out)
            self.layers.append(l)
        outputs = Layer(self.hidden_dim, self.output_dim)
        self.layers.append(outputs)

    @staticmethod
    def f_activation(z):
        act = np.vectorize(lambda _z : 1 / (1 + np.exp(-1*_z)))
        return act(z)

    def draw(self):
        print ' '.join(['I' for x in range(self.input_dim)])
        for l in self.layers:
            print ' '.join(['X' for x in range(l.countOuputs())])

    def getSingleOutput(self, X):
    	'''
    	Given input vector X
    	'''
        a = X
        for l in self.layers:
            a_new = l.forward(a)
            a = a_new
        return a

class Layer():

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(self.input_dim, self.output_dim)  # Weight tensor (output of each neuron)
        self.b = np.random.randn(1, self.output_dim)  # bias (size = output_dim)

    def countOuputs(self):
        return self.output_dim

    def forward(self, X):
        '''
        Forward propagation
        X is inputs vector to pass into layer, size = input_dim
        Returns activations a
        '''
        z = np.dot(X, self.W) + self.b
        a = Brain.f_activation(z)
        return a


