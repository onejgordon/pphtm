#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pphtm.pphtm_brain import PPHTMBrain
from chtm.chtm_printer import CHTMPrinter
from pphtm.pphtm_predictor import PPHTMPredictor
import numpy as np
import random
from encoders import SimpleFullWidthEncoder

END_FILE_PCT = .25

# values are either tuple of range (min, max) inclusive, or static value
# ranges are considered integer ranges if min is an integer, otherwise float
SWARM_CONFIG = {
    'PROXIMAL_ACTIVATION_THRESHHOLD': 3,
    'DISTAL_ACTIVATION_THRESHOLD': 2,
    'BOOST_MULTIPLIER': (1.2, 1.4),
    'DESIRED_LOCAL_ACTIVITY': 2,
    'DO_BOOSTING': 1,
    'DISTAL_SYNAPSE_CHANCE': 0.5,
    'TOPDOWN_SYNAPSE_CHANCE': 0.4,
    'MAX_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.4,
    'MIN_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.1
}

class RunResult(object):

    def __init__(self, iteration_id=0, params=None, percent_correct=0.0, percent_correct_end=0.0):
        self.iteration_id = iteration_id
        self.params = params # dict
        self.percent_correct = percent_correct
        self.percent_correct_end = percent_correct_end

    def __str__(self):
        pretty_good = self.percent_correct_end > 0.5
        out = "Run Result (%d) %% correct: %.1f, %% correct end: %.1f" % (self.iteration_id, 100.0 * self.percent_correct, 100.0 * self.percent_correct_end)
        if pretty_good:
            out += " <<< GOOD"
        return out

    def print_params(self):
        for k, v in self.params.items():
            if SWARM_CONFIG[k] != v:
                print "%s -> %s" % (k, v)

class SwarmRunner(object):

    DATA_DIR = "data"
    ALPHA = "ABCDEF"
    CROP_FILE = 180
    N_INPUTS = 36
    CPR = [9**2] # Cells per region

    def __init__(self, filename="simple_pattern2.txt", iterations=5):
        self.b = PPHTMBrain(cells_per_region=self.CPR, min_overlap=1, r1_inputs=self.N_INPUTS)
        self.b.initialize()
        self.classifier = PPHTMPredictor(self.b, categories=self.ALPHA)
        self.filename = filename
        self.encoder = SimpleFullWidthEncoder(n_inputs=self.N_INPUTS, n_cats=len(self.ALPHA))
        self.iterations = iterations
        self.data = None
        self.cursor = 0 # data read index
        self.iteration_index = 0
        self.results = {} # iteration index -> RunResult()

    def _encode_letter(self, c):
        i = ord(c.upper()) - 65 # A == 0
        return self.encoder.encode(i)

    def _read_data(self):
        with open(self.DATA_DIR + "/" + self.filename, 'r') as myfile:
            self.data = myfile.read()

    def _choose_params(self):
        params = {}
        for key, spec in SWARM_CONFIG.items():
            if type(spec) in [int, float]:
                params[key] = spec # static
            elif type(spec) is tuple:
                _min, _max = spec # unpack
                if type(_min) is int:
                    # integer range
                    params[key] = random.randint(_min, _max)
                elif type(_min) is float:
                    # continuous range
                    params[key] = random.uniform(_min, _max)
        return params

    def run(self):
        self._read_data()
        for i in range(self.iterations):
            params = self._choose_params()
            # Initialize brain with selected params
            self.b.initialize(**params)
            done = False
            prediction = None
            correct_predictions = 0
            correct_predictions_end = 0
            self.cursor = 0
            self.classifier.initialize()
            while not done:
                self.cursor += 1
                char = self.data[self.cursor].upper()
                print "Pred: %s, char: %s" % (prediction, char)
                if prediction and char == prediction:
                    correct_predictions += 1
                    if (float(self.cursor) / self.CROP_FILE) > (1.0 - END_FILE_PCT):
                        # Last 25%
                        correct_predictions_end += 1
                prediction = self.process(char)
                done = self.cursor >= self.CROP_FILE
            pct_correct = float(correct_predictions) / self.CROP_FILE
            pct_correct_end = float(correct_predictions_end) / (END_FILE_PCT * self.CROP_FILE)
            self.results[i] = RunResult(iteration_id=i, params=params, percent_correct=pct_correct, percent_correct_end=pct_correct_end)
            print "Iteration %d done" % (i)

        print "Swarm run done!"
        for iteration_id, runresult in self.results.items():
            print runresult
            runresult.print_params()

    def process(self, char):
        # Process one step
        inputs = self._encode_letter(char)
        self.b.process(inputs, learning=True)
        self.classifier.read(char)
        prediction = self.classifier.predict()
        return prediction

def main():
    swarm = SwarmRunner(iterations=10)
    swarm.run()

if __name__ == "__main__":
    main()
