#!/usr/bin/env python

import math
import util
import numpy as np

CONSIDERATION_THRESHOLD = 1.0
CLASSIFY_ON = "activation" # or bias

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
        self.value_history = np.zeros((0, self.region.n_cells))
        self.t = 0

    def read(self, input):
        '''
        Input is raw category value index
        '''
        if CLASSIFY_ON == "activation":
            values = [c.activation for c in self.region.cells]
        elif CLASSIFY_ON == "bias":
            values = self.region.bias
        self.input_history.append(input) # Add new input
        self.value_history = np.vstack((self.value_history, values)) # Add new values
        if len(self.input_history) > self.history_window:
            self.input_history = self.input_history[1:]
            self.value_history = self.value_history[1:]

    def predict(self):
        # Build tally list (number of times each cat came on k-steps after each cell)
        # CONFIRM: bias classification predicts this time step (bias not updated until next input)
        k = 1 if CLASSIFY_ON == "activation" else 0
        self.tallies = [np.zeros(len(self.categories)) for x in range(self.region.n_cells)]
        for t, value in enumerate(self.value_history):
            if t+k < len(self.input_history):
                input_t_plus_k = self.input_history[t+k]
                if input_t_plus_k in self.categories:
                    high_indexes = np.where(value >= CONSIDERATION_THRESHOLD)[0]
                    for index in high_indexes:
                        cat_index = self.categories.find(input_t_plus_k)
                        self.tallies[index][cat_index] += 1

        if CLASSIFY_ON == "activation":
            current_values = np.array([cell.activation for cell in self.region.cells])
        elif CLASSIFY_ON == "bias":
            current_values = self.region.bias

        high_indexes = np.where(current_values >= CONSIDERATION_THRESHOLD)[0]
        predicted_distribution = np.zeros(len(self.categories))
        for index in high_indexes:
            cell_probability = self.tallies[index]
            predicted_distribution += cell_probability

        normalized = predicted_distribution / sum(predicted_distribution)
        predicted_index = np.argmax(normalized)
        return self.categories[predicted_index]
