#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pphtm.pphtm_brain import PPHTMBrain
from pphtm.pphtm_predictor import PPHTMPredictor
from datetime import datetime
import numpy as np
import util
import random
from encoders import SimpleFullWidthEncoder
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

VERBOSITY = 1
GOOD_CUTOFF_PCT = 0.6
FILENAME = "longer_char_sequences1.txt"
ALPHA = "ABCDEFG" # All data in file (auto-produce?)
# FILENAME = "simple_pattern2.txt"
# ALPHA = "ABCDEF"
ITERATIONS = 30
RERUN_PARAMS = 1 # No. of runs for each randomized param set
CROP_FILE = 150
END_FILE_PCT = .15

# values are either:
# - tuple of range (min, max) inclusive,
# - list of options (choose one), or
# - static value
# ranges are considered integer ranges if min is an integer, otherwise continuous
SWARM_CONFIG = {
    'PROXIMAL_ACTIVATION_THRESHHOLD': 3,
    'DISTAL_ACTIVATION_THRESHOLD': 2,
    'BOOST_MULTIPLIER': 2.58, #(2.0, 3.0),
    'DESIRED_LOCAL_ACTIVITY': 2,
    'DISTAL_SYNAPSE_CHANCE': 0.5,
    'TOPDOWN_SYNAPSE_CHANCE': 0.5,
    'MAX_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.6,
    'MIN_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.1,
    'CELLS_PER_REGION': 9**2, #[6**2, 8**2, 10**2],
    'N_REGIONS': 1,
    'BIAS_WEIGHT': 1.0,
    'OVERLAP_WEIGHT': 0.6,
    'FADE_RATE': 0.7,
    'DISTAL_SEGMENTS': 2,
    'PROX_SEGMENTS': 2,
    'TOPDOWN_SEGMENTS': 2, # Only relevant if >1 region
    'SYNAPSE_DECAY': 0.0008,
    'INIT_PERMANENCE_LEARN_INC_CHANGE': 0.03,
    'INIT_PERMANENCE_LEARN_DEC_CHANGE': 0.003,
    'CHANCE_OF_INHIBITORY': 0.2,
    'SYNAPSE_ACTIVATION_LEARN_THRESHHOLD': 1.0,
    'DISTAL_BOOST_MULT': 0.02,
    'INHIBITION_RADIUS_DISCOUNT': 0.8,
    # Booleans
    'DO_BOOSTING': 1
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
        return self.percent_correct_end > GOOD_CUTOFF_PCT # TODO: customize based on ave word len

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

    def _variable_params(self):
        params = []
        for k, v in SWARM_CONFIG.items():
            if type(v) in [tuple, list]:
                # variable
                params.append(k)
        params = sorted(params) # Alpha sort for consistency
        return params

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
        param_run_count = 1
        for i in range(self.iterations):
            print "Running iteration %d/%d" % (i, self.iterations)
            if self.params and param_run_count < RERUN_PARAMS:
                print "Param run %d" % param_run_count
                param_run_count += 1
            else:
                param_run_count = 1
                self._choose_params()

            # Initialize brain with selected params
            self.b.initialize(**self.params)
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
                correct_prediction = prediction and char == prediction
                if correct_prediction:
                    correct_predictions += 1
                    if (float(self.cursor) / CROP_FILE) > (1.0 - END_FILE_PCT):
                        correct_predictions_end += 1
                prediction = self.process(char)
                out_char = "P" if correct_prediction else "."
                log(out_char, level=2)
                done = self.cursor >= CROP_FILE
            pct_correct = float(correct_predictions) / CROP_FILE
            pct_correct_end = float(correct_predictions_end) / (END_FILE_PCT * CROP_FILE)
            rr = RunResult(iteration_id=i, params=self.params, percent_correct=pct_correct, percent_correct_end=pct_correct_end)
            self.results[i] = rr
            log("Iteration %d/%d done - %s" % (i, self.iterations, rr))
            log(rr.print_params())

        self.end_time = datetime.now()
        log("Swarm run done!")
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
            text_file.write(">>> %.1f <<< Percent of runs good" % (float(n_good)*100./self.iterations))
        self.plot_results()

    def process(self, char):
        # Process one step
        inputs = self._encode_letter(char)
        self.b.process(inputs, learning=True)
        self.classifier.read(char)
        prediction = self.classifier.predict()
        return prediction

    def plot_results(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_params = self._variable_params()[:3] # First 3 variable params
        n = 100
        xs = []
        ys = []
        zs = []
        colors = []
        title = 'Summary of swarm runs'
        for i, pp in enumerate(plot_params):
            if i == 0:
                ax.set_xlabel(pp)
            elif i == 1:
                ax.set_ylabel(pp)
            elif i == 2:
                title += " (color = %s)" % pp
        for index, rr in self.results.items():
            for i, pp in enumerate(plot_params):
                if i == 0:
                    xs.append(rr.params.get(pp))
                elif i == 1:
                    ys.append(rr.params.get(pp))
                elif i == 2:
                    colors.append(rr.params.get(pp))
            zs.append(rr.percent_correct_end) # Z always percent correct (outcome rating)
        if xs and ys and zs:
            if colors:
                ax.scatter(xs, ys, zs, c=colors, cmap=cm.coolwarm)
            else:
                ax.scatter(xs, ys, zs)
            ax.set_zlabel('Percent Correct at End (%)')
            plt.title(title)
            plt.show()
        else:
            print "Can't plot, not enough dimensions"

def log(message, level=1):
    if VERBOSITY >= level:
        print message


def main():
    swarm = SwarmRunner(FILENAME, iterations=ITERATIONS)
    swarm.run()

if __name__ == "__main__":
    main()
