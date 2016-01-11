#!/usr/bin/env python

import math
import util
import numpy as np

class CHTMClassifier(object):

    ON_RATE_THRESH = 0.7

    def __init__(self, brain, categories=None, region_index=None, history_window=500):
        self.brain = brain
        self.region_index = region_index
        self.history_window = history_window
        self.categories = categories
        self.region = brain.regions[self.region_index]
        # Tallies sublist is histogram of counts for each category. Main list is each bit/cell
        self.tallies = None
        self.input_history = [] # Len == history window
        self.activation_history = np.zeros((0, self.region.n_cells))
        self.t = 0

    def read(self, input):
        '''
        Input is raw category value index
        '''
        activations = [c.activation for c in self.brain.regions[self.region_index].cells]
        self.input_history.append(input) # Add new input
        self.activation_history = np.vstack((self.activation_history, activations)) # Add new activations
        if len(self.input_history) > self.history_window:
            self.input_history = self.input_history[1:]
            self.activation_history = self.activation_history[1:]            

    def predict(self, k=1):
        '''
        k is distance into future to predict
        '''
        # Build tally list
        self.tallies = [np.zeros(len(self.categories)) for x in range(self.region.n_cells)]
        for t, activation in enumerate(self.activation_history):
            if t+k < len(self.input_history):
                input_t_plus_k = self.input_history[t+k]
                if input_t_plus_k in self.categories:
                    active_indexes = np.where(activation == 1.0)[0]
                    for index in active_indexes:
                        cat_index = self.categories.find(input_t_plus_k)
                        self.tallies[index][cat_index] += 1

        current_activation = np.array([cell.activation for cell in self.region.cells])

        active_indexes = np.where(current_activation == 1.0)[0]
        predicted_distribution = np.zeros(len(self.categories))
        for index in active_indexes:
            cell_probability = self.tallies[index]
            predicted_distribution += cell_probability

        normalized = predicted_distribution / sum(predicted_distribution)
        predicted_index = np.argmax(normalized)
        return self.categories[predicted_index]
