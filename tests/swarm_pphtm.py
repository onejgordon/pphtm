#!/usr/bin/env python
import sys, getopt
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pphtm.pphtm_brain import PPHTMBrain
from pphtm.pphtm_predictor import PPHTMPredictor
from datetime import datetime
import numpy as np
import util
import random
from encoders import SimpleFullWidthEncoder
from helpers.file_processer import FileProcesser
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

VERBOSITY = 1
GOOD_CUTOFF_PCT = 0.75
RERUN_PARAMS = 3 # No. of runs for each randomized param set
END_FILE_PCT = .15
PREDICTION_STREAK_WINDOW = 80

# values are either:
# - tuple of range (min, max) inclusive,
# - list of options (choose one), or
# - static value
# ranges are considered integer ranges if min is an integer, otherwise continuous
SWARM_CONFIG = {
    # 'PROXIMAL_ACTIVATION_THRESHHOLD': 3,
    # 'DISTAL_ACTIVATION_THRESHOLD': 2,
    # 'BOOST_MULTIPLIER': (1.0, 2.9), # 2.58
    # 'DESIRED_LOCAL_ACTIVITY': 2,
    # 'DISTAL_SYNAPSE_CHANCE': 0.5,
    # 'TOPDOWN_SYNAPSE_CHANCE': 0.5,
    # 'MAX_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.6,
    # 'MIN_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.1,
    # 'CELLS_PER_REGION': 14**2,
    # 'N_REGIONS': 2,
    # 'BIAS_WEIGHT': (0.5, 0.9),
    # 'OVERLAP_WEIGHT': (0.4, 0.6),
    # 'FADE_RATE': (0.2, 0.6),
    'DISTAL_SEGMENTS': [2, 3, 4],
    # 'PROX_SEGMENTS': 2,
    # 'TOPDOWN_SEGMENTS': 2, # Only relevant if >1 region
    # 'SYNAPSE_DECAY_PROX': 0.0008,
    # 'SYNAPSE_DECAY_DIST': 0.0008,
    # 'PERM_LEARN_INC': 0.08,
    # 'PERM_LEARN_DEC': 0.05,
    'CHANCE_OF_INHIBITORY': [0.0, 0.1],
    # 'SYNAPSE_ACTIVATION_LEARN_THRESHHOLD': 1.0,
    # 'DISTAL_BOOST_MULT': 0.02,
    # 'INHIBITION_RADIUS_DISCOUNT': 0.8,
    # Booleans
}


class RunResult(object):

    def __init__(self, iteration_id=0, params=None, percent_correct=0.0, percent_correct_end=0.0, streak_max=0.0):
        self.iteration_id = iteration_id
        self.params = params # dict
        self.percent_correct = percent_correct
        self.percent_correct_end = percent_correct_end
        self.streak_max = streak_max

    def __str__(self):
        out = "Run Result (%d): %% correct end: %.1f, max (%d step window): %.1f" % (self.iteration_id + 1, 100.0 * self.percent_correct_end, PREDICTION_STREAK_WINDOW, 100.0 * self.streak_max)
        if self.good():
            out += " <<< GOOD"
        return out

    def good(self):
        '''Good run, as defined by max streak percent
        '''
        # TODO: customize based on ave word len
        if True:
            return self.streak_max >= GOOD_CUTOFF_PCT
        else:
            return self.percent_correct_end >= GOOD_CUTOFF_PCT

    def print_params(self, unique_only=True):
        res = ""
        for k, v in self.params.items():
            if SWARM_CONFIG[k] != v or not unique_only:
                res += "%s -> %s\n" % (k, v)
        return res

class SwarmRunner(object):

    DATA_DIR = "../data"

    def __init__(self, filename=None, iterations=5, crop=100):
        self.file_processer = FileProcesser(filename=filename)
        self.cats, self.data, self.n_inputs = self.file_processer.open_file()
        self.b = PPHTMBrain(min_overlap=1, r1_inputs=self.n_inputs)
        self.b.initialize()
        self.classifier = PPHTMPredictor(self.b, categories=self.cats)
        self.encoder = SimpleFullWidthEncoder(n_inputs=self.n_inputs, n_cats=len(self.cats))
        self.iterations = iterations
        self.crop = crop
        self.iteration_index = 0
        self.params = {}
        self.results = {} # iteration index -> RunResult()
        self.start_time, self.end_time = None, None

    def _print_static_params(self):
        res = ""
        static_config = dict(self.b.CONFIG) # Get full brain config
        for k, v in static_config.items():
            swarm_param = SWARM_CONFIG.get(k)
            if swarm_param and type(swarm_param) not in [int, float]:
                # variable, remove from static config
                del static_config[k]
        for k, v in static_config.items():
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
        res += "File: %s\n" % self.file_processer.filename
        res += "N Inputs: %s\n" % self.n_inputs
        res += "Started: %s\n" % util.sdatetime(self.start_time)
        res += "Finished: %s\n" % util.sdatetime(self.end_time)
        res += "Duration: %s\n" % util.duration(self.start_time, self.end_time)
        res += "Crop file: %s, End file percent: %s\n" % (self.crop, END_FILE_PCT)
        res += "Common Specs:\n"
        res += self._print_static_params()
        res += "------------------------\n\n"
        return res

    def run(self):
        self.start_time = datetime.now()
        param_run_count = 1
        for i in range(self.iterations):
            print "Running iteration %d of %d (crop file: %s)" % (i+1, self.iterations, self.crop)
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
            self.file_processer.reset()
            self.classifier.initialize()
            prediction_streak = [] # max len of PREDICTION_STREAK_WINDOW
            streak_max = 0.0
            while not done:
                char = self.file_processer.read_next()
                # print "Pred: %s, char: %s" % (prediction, char)
                correct_prediction = prediction and char == prediction
                if correct_prediction:
                    correct_predictions += 1
                    if (float(self.file_processer.cursor) / self.crop) > (1.0 - END_FILE_PCT):
                        correct_predictions_end += 1
                prediction = self.process(char)
                util.rolling_list(prediction_streak, length=PREDICTION_STREAK_WINDOW, new_value=correct_prediction)
                if len(prediction_streak) >= PREDICTION_STREAK_WINDOW:
                    streak_percent = util.average(prediction_streak)
                    if streak_percent > streak_max:
                        streak_max = streak_percent
                done = self.file_processer.cursor >= self.crop
            pct_correct = float(correct_predictions) / self.crop
            pct_correct_end = float(correct_predictions_end) / (END_FILE_PCT * self.crop)
            rr = RunResult(iteration_id=i, params=self.params, percent_correct=pct_correct, percent_correct_end=pct_correct_end, streak_max=streak_max)
            self.results[i] = rr
            log("Iteration %d/%d done - %s" % (i, self.iterations, rr))
            log(rr.print_params())

        self.end_time = datetime.now()
        log("Swarm run done!")
        sorted_results = sorted(self.results.items(), key=lambda r : r[1].streak_max)
        n_good = sum([rr[1].good() for rr in sorted_results])
        fname = "Swarm at %s %d of %d runs good" % (util.sdatetime(self.start_time), n_good, len(sorted_results))
        with open("../outputs/%s.txt" % fname, "w") as text_file:
            text_file.write(self._run_header())
            for iteration_id, runresult in sorted_results:
                print runresult
                print runresult.print_params()
                text_file.write(str(runresult)+"\n")
                text_file.write(runresult.print_params()+"\n")
            text_file.write(">>> %.1f <<< Percent of runs good" % (float(n_good)*100./self.iterations))
        if self.iterations > 1:
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
            zs.append(rr.streak_max) # Z always streak max (outcome rating)
        if xs and zs:
            outcome_label = 'Max Streak Percent Correct, %d step window (%%)' % PREDICTION_STREAK_WINDOW
            chart_2d = len(plot_params) == 1
            if not chart_2d:
                ax.set_zlabel(outcome_label)
            else:
                # Just charting our one variable against correct (2D)
                ax.set_ylabel(outcome_label)
                ys = 0
            if colors:
                # 4d
                ax.scatter(xs, ys, zs, c=colors, cmap=cm.coolwarm)
            else:
                if chart_2d:
                    ax.scatter(xs, zs, zs=0, zdir='y')
                else:
                    # 3d
                    ax.scatter(xs, ys, zs)
            plt.title(title)
            plt.show()
        else:
            print "Not plotting, not enough dimensions"

def log(message, level=1):
    if VERBOSITY >= level:
        print message


def main(argv):
    HELP = 'swarm_pphtm.py -i <iterations> -c <crop> -f <file>'
    try:
        opts, args = getopt.getopt(argv,"hi:c:",["iterations=","crop="])
    except getopt.GetoptError:
        print HELP
        sys.exit(2)
    # Defaults
    kwargs = {
        'iterations': 30,
        'crop': 300,
        'filename': "longer_char_sequences2.txt"
    }
    for opt, arg in opts:
        if opt == '-h':
            print HELP
            sys.exit()
        elif opt in ("-i", "--iterations"):
            kwargs['iterations'] = int(arg)
        elif opt in ("-c", "--crop"):
            kwargs['crop'] = int(arg)
        elif opt in ("-f", "--file"):
            kwargs['filename'] = arg
    swarm = SwarmRunner(**kwargs)
    swarm.run()

if __name__ == "__main__":
   main(sys.argv[1:])