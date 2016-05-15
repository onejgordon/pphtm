#!/usr/bin/env python

import math
import util
import numpy as np
import operator

CONSIDERATION_THRESHOLD = 0.5
from encoders import SimpleFullWidthEncoder

class PPHTMPredictor(object):
    '''
    Take present activation of region 0 and follow proximal connections
    down to predict sensory inputs
    '''

    def __init__(self, brain, categories):
        self.brain = brain
        self.categories = categories
        self.region = brain.regions[0]
        self.overlap_lookup = {} # raw_input (e.g. letter) -> overlap np.array
        # Tallies counts number of 'votes' for each input based on proximal connections
        # of r0 biased cells
        self.encoder = SimpleFullWidthEncoder(n_inputs=self.region.n_inputs, n_cats=len(categories))

    def predict_via_overlap_lookup(self):
        '''
        Compare raw input -> overlap map to look for best match with bias
        '''
        scores = {}
        for raw_input, overlap in self.overlap_lookup.items():
            score = util.bool_overlap(overlap, self.region.bias)
            scores[raw_input] = score

        max_raw_input, max_score = max(scores.iteritems(), key=operator.itemgetter(1))
        return max_raw_input


    def predict_via_proximal_reversal(self):
        '''
        Reverse biased cells down proximal segments and compare against encoded input options
        '''
        self.tallies = np.zeros(self.region.n_inputs)
        for i, bias in enumerate(self.region.bias):
            if bias >= CONSIDERATION_THRESHOLD:
                cell = self.region.cells[i]
                for seg in cell.proximal_segments:
                    for source_index in seg.syn_sources:
                        self.tallies[source_index] += 1

        overlap_scores = []
        for cat in self.categories:
            i = ord(cat.upper()) - 65 # A == 0
            encoded = self.encoder.encode(i)
            overlap_scores.append(self.overlap(encoded, self.tallies))

        max_overlap_score = max(overlap_scores)
        predicted_index = overlap_scores.index(max_overlap_score)
        return self.categories[predicted_index]

    def predict(self):
        if True:
            return self.predict_via_overlap_lookup()
        else:
            return self.predict_via_proximal_reversal()

    def overlap(self, input, tallies):
        '''
        Compare real encoded inputs to tallies

        Returns:
            Overlap score
        '''
        return util.bool_overlap(input, tallies)

    def read(self, raw_input):
        if raw_input:
            self.overlap_lookup[raw_input] = np.copy(self.region.overlap)

    def overlap_lookup_draw_fn(self, raw_input, index):
        last_overlap_for_input = self.overlap_lookup.get(raw_input)
        value = 0
        if last_overlap_for_input is not None:
            value = last_overlap_for_input[index]
        ind_color = ol_color = None
        bias = self.region.bias[index]
        if value and bias:
            ind_color = "#FF0000"
        return (value, ind_color, ol_color)

