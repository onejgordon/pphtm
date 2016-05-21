#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pphtm.pphtm_brain import PPHTMBrain
from chtm.chtm_printer import CHTMPrinter
from pphtm.pphtm_predictor import PPHTMPredictor
from datetime import datetime
import numpy as np
import util
import random
from encoders import SimpleFullWidthEncoder

FILENAME = "longer_char_sequences1.txt"
ALPHA = "ABCDEFG" # All data in file (auto-produce?)
ITERATIONS = 10
CROP_FILE = 300
END_FILE_PCT = .15

# values are either:
# - tuple of range (min, max) inclusive,
# - list of options (choose one), or
# - static value
# ranges are considered integer ranges if min is an integer, otherwise continuous
SWARM_CONFIG = {
    'PROXIMAL_ACTIVATION_THRESHHOLD': (2,3),
    'DISTAL_ACTIVATION_THRESHOLD': (2, 3),
    'BOOST_MULTIPLIER': (1.5, 2.1),
    'DESIRED_LOCAL_ACTIVITY': 2,
    'DO_BOOSTING': 1,
    'DISTAL_SYNAPSE_CHANCE': 0.5,
    'TOPDOWN_SYNAPSE_CHANCE': 0.4,
    'MAX_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.4,
    'MIN_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.1,
    'CELLS_PER_REGION': [7**2, 8**2, 9**2],
    'N_REGIONS': [1,2]
}

class RunResult(object):

    def __init__(self, iteration_id=0, params=None, percent_correct=0.0, percent_correct_end=0.0):
        self.iteration_id = iteration_id
        self.params = params # dict
        self.percent_correct = percent_correct
        self.percent_correct_end = percent_correct_end

    def __str__(self):
        out = "Run Result (%d) %% correct: %.1f, %% correct end: %.1f" % (self.iteration_id, 100.0 * self.percent_correct, 100.0 * self.percent_correct_end)
        if self.good():
            out += " <<< GOOD"
        return out

    def good(self):
        return self.percent_correct_end > 0.6 # TODO: customize based on ave word len

    def print_params(self, unique_only=True):
        res = ""
        for k, v in self.params.items():
            if SWARM_CONFIG[k] != v or not unique_only:
                res += "%s -> %s\n" % (k, v)
        return res

class SwarmRunner(object):

    DATA_DIR = "../data"
    N_INPUTS = 7**2

    def __init__(self, filename, iterations=5):
        self.b = PPHTMBrain(min_overlap=1, r1_inputs=self.N_INPUTS)
        self.b.initialize()
        self.classifier = PPHTMPredictor(self.b, categories=ALPHA)
        self.filename = filename
        self.encoder = SimpleFullWidthEncoder(n_inputs=self.N_INPUTS, n_cats=len(ALPHA))
        self.iterations = iterations
        self.data = None
        self.cursor = 0 # data read index
        self.iteration_index = 0
        self.params = {}
        self.results = {} # iteration index -> RunResult()
        self.start_time, self.end_time = None, None

    def _print_static_params(self):
        res = ""
        for k, v in SWARM_CONFIG.items():
            if type(v) in [int, float]:
                # static
                res += "%s -> %s\n" % (k, v)
        return res

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
            elif type(spec) is list:
                # Choose one
                params[key] = random.choice(spec)
            elif type(spec) is tuple:
                # Treat as range (int or continuous)
                _min, _max = spec # unpack
                if type(_min) is int:
                    # integer range
                    params[key] = random.randint(_min, _max)
                elif type(_min) is float:
                    # continuous range
                    params[key] = random.uniform(_min, _max)
        self.params = params
        return params

    def _run_header(self):
        res = ""
        res += "File: %s\n" % self.filename
        res += "N Inputs: %s\n" % self.N_INPUTS
        res += "Started: %s\n" % util.sdatetime(self.start_time)
        res += "Finished: %s\n" % util.sdatetime(self.end_time)
        res += "Duration: %s\n" % util.duration(self.start_time, self.end_time)
        res += "Crop file: %s, End file percent: %s\n" % (CROP_FILE, END_FILE_PCT)
        res += "Common Specs:\n"
        res += self._print_static_params()
        res += "------------------------\n\n"
        return res

    def run(self):
        self._read_data()
        self.start_time = datetime.now()
        for i in range(self.iterations):
            print "Running iteration %d" % (i)
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
                # print "Pred: %s, char: %s" % (prediction, char)
                if prediction and char == prediction:
                    correct_predictions += 1
                    if (float(self.cursor) / CROP_FILE) > (1.0 - END_FILE_PCT):
                        # Last 25%
                        correct_predictions_end += 1
                prediction = self.process(char)
                done = self.cursor >= CROP_FILE
            pct_correct = float(correct_predictions) / CROP_FILE
            pct_correct_end = float(correct_predictions_end) / (END_FILE_PCT * CROP_FILE)
            self.results[i] = RunResult(iteration_id=i, params=params, percent_correct=pct_correct, percent_correct_end=pct_correct_end)
            print "Iteration %d done" % (i)

        self.end_time = datetime.now()
        print "Swarm run done!"
        sorted_results = sorted(self.results.items(), key=lambda r : r[1].percent_correct_end)
        fname = "Run at " + util.sdatetime(self.start_time)
        n_good = 0
        with open("../outputs/%s.txt" % fname, "w") as text_file:
            text_file.write(self._run_header())
            for iteration_id, runresult in sorted_results:
                print runresult
                print runresult.print_params()
                text_file.write(str(runresult)+"\n")
                text_file.write(runresult.print_params()+"\n")
                if runresult.good():
                    n_good += 1
            text_file.write(">>> %.0f <<< Percent of runs good" % (float(n_good)/len(self.results.keys()) ))

    def process(self, char):
        # Process one step
        inputs = self._encode_letter(char)
        self.b.process(inputs, learning=True)
        self.classifier.read(char)
        prediction = self.classifier.predict()
        return prediction

def main():
    swarm = SwarmRunner(FILENAME, iterations=ITERATIONS)
    swarm.run()

if __name__ == "__main__":
    main()
