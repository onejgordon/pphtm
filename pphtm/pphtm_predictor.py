#!/usr/bin/env python

import math
from pphtm import util
import numpy as np
import operator

CONSIDERATION_THRESHOLD = 0.5
from pphtm.encoders import SimpleFullWidthEncoder

class PPHTMPredictor(object):
    '''
    Take present activation of region 0 and follow proximal connections
    down to predict sensory inputs
    '''

    def __init__(self, brain, categories):
        self.brain = brain
        self.categories = categories
        self.region = None
        self.overlap_lookup = {} # raw_input (e.g. letter) -> overlap np.array
        self.activation_lookup = {} # sequence (e.g. "BA") -> activation np.array
        # Tallies counts number of 'votes' for each input based on proximal connections
        # of r0 biased cells
        self.encoder = SimpleFullWidthEncoder(n_inputs=self.brain.n_inputs, n_cats=len(categories))

    def initialize(self):
        self.region = self.brain.regions[0]
        self.overlap_lookup = {}
        self.activation_lookup = {}

    def predict_via_overlap_lookup(self):
        '''
        Compare raw input -> overlap map to look for best match with bias
        TODO: Should we consider both active and inactive matches?
        '''
        scores = {}
        for raw_input, overlap in self.overlap_lookup.items():
            # score = util.bool_overlap(overlap, self.region.bias) / sum(overlap)
            score = np.dot(overlap, self.region.bias) / sum(overlap)
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

    def read(self, raw_input, prior_input=None):
        if raw_input:
            self.overlap_lookup[raw_input] = np.copy(self.region.overlap)
            if prior_input:
                key = prior_input + raw_input
                self.activation_lookup[key] = np.copy(self.region.activation == 1)

