#!/usr/bin/env python
import sys
from os import path
sys.path.append('/Users/jeremygordon/pyprojects')
from pphtm.pphtm.pphtm_brain import PPHTMBrain, Segment, Cell
from pphtm.chtm.chtm_printer import CHTMPrinter
from pphtm.pphtm.pphtm_predictor import PPHTMPredictor
from pphtm.helpers.file_processer import FileProcesser

from nupic.encoders.scalar import ScalarEncoder
from pphtm.encoders import SimpleFullWidthEncoder
import random
import numpy as np
import math

USE_SIMPLE_ENCODER = False
SHOW_RUN_SUMMARY = False


class DesignedUserInputRunner(object):
    '''
    Runs a single data file through a PPHTM and visualizes processing.
    '''

    def __init__(self, filename=None, animate=True):
        self.SEQUENCES = [
            "BACD",
            "ABCD",
            "DAAC"
        ]
        self.LETTERS = "ABCD"
        self.PROX_ACTIVATION = 10
        self.file_processer = FileProcesser(filename="spatial_sequences.txt", with_sequences=True)
        self.cats, self.data, self.n_inputs = self.file_processer.open_file()
        self.n_inputs = 11**2
        self.CPR = 14**2
        self.classifier = None
        self.current_batch_target = 0
        self.current_batch_counter = 0
        self.seq_matches = {}  # sequence index -> dict ('correct', 'incorrect') counts of false-pos false-neg
        config = {
            "PROXIMAL_ACTIVATION_THRESHHOLD": self.PROX_ACTIVATION,
            # "DISTAL_ACTIVATION_THRESHOLD": 2,
            # "BOOST_MULTIPLIER": 2.58,
            # "DESIRED_LOCAL_ACTIVITY": 2,
            # "DISTAL_SYNAPSE_CHANCE": 0.4,
            # "TOPDOWN_SYNAPSE_CHANCE": 0.3,
            # "MAX_PROXIMAL_INIT_SYNAPSE_CHANCE": 0.4,
            # "MIN_PROXIMAL_INIT_SYNAPSE_CHANCE": 0.1,
            "CELLS_PER_REGION": self.CPR,
            "N_REGIONS": 2,
            "BIAS_WEIGHT": 0.0,
            "OVERLAP_WEIGHT": 1.0,
            # "FADE_RATE": 0.5,
            # None to start, we will manually add
            "DISTAL_SEGMENTS": 0,
            "PROX_SEGMENTS": 0,
            "TOPDOWN_SEGMENTS": 0,
            # "TOPDOWN_BIAS_WEIGHT": 0.5,
            # "SYNAPSE_DECAY_PROX": 0.00005,
            # "SYNAPSE_DECAY_DIST": 0.0,
            # "PERM_LEARN_INC": 0.07,
            # "PERM_LEARN_DEC": 0.04,
            # "CHANCE_OF_INHIBITORY": 0.1,
            # "DIST_SYNAPSE_ACTIVATION_LEARN_THRESHHOLD": 1.0,
            # "PROX_SYNAPSE_ACTIVATION_LEARN_THRESHHOLD": 0.5,
            # "DISTAL_BOOST_MULT": 0.01,
            # "INHIBITION_RADIUS_DISCOUNT": 0.8
        }

        if USE_SIMPLE_ENCODER:
            self.encoder = SimpleFullWidthEncoder(n_inputs=self.n_inputs, n_cats=len(self.LETTERS))
        else:
            self.encoder = ScalarEncoder(n=self.n_inputs, w=int(math.sqrt(self.n_inputs)), minval=0, maxval=len(self.LETTERS)-1, periodic=False, forced=True)

        self.b = PPHTMBrain(min_overlap=1, r1_inputs=self.n_inputs)
        self.b.initialize(**config)
        self.design()

        self.animate = animate
        self.delay = 50
        self.char = None
        self.quitting = False

        self.printer = CHTMPrinter(self.b, handle_run_batch=self.start_batch, handle_quit=self.quit)
        self.printer.setup()

    def design(self):
        # Overwrite connections and weights to construct certain design
        # We want a network with a simple output region (one cell per sequence)
        # We assign a pattern in region 1 for each letter
        # Each letter's pattern has a core set and a peripheral set
        mapping = {}  # char -> dict (lists: 'core', 'periph_exc', 'periph_inh')
        EXC_PERIPHERAL_PROB = 0.5
        SLOW_FADE_RATE = 0.3
        FAST_FADE_RATE = 0.6
        r2 = self.b.regions[0]
        for i, char in enumerate(self.LETTERS):
            mapping[char] = {'core': [], 'periph_exc': [], 'periph_inh': []}
            core_r2 = random.sample(r2.cells, self.CPR/5)
            peripheral_r2 = random.sample(r2.cells, self.CPR/4)
            encoded_input = self.encoder.encode(i)
            for cell in core_r2:
                proximal = Segment(cell, 0, r2, type=Segment.PROXIMAL)
                sources = random.sample(np.where(encoded_input)[0], self.PROX_ACTIVATION)
                for source in sources:
                    proximal.add_synapse(source, permanence=0.3)
                cell.proximal_segments = [proximal]
                cell.excitatory = True
                cell.fade_rate = SLOW_FADE_RATE
                mapping[char]['core'].append(cell.index)
            for cell in peripheral_r2:
                if not cell.proximal_segments:
                    excitatory = random.random() < EXC_PERIPHERAL_PROB
                    proximal = Segment(cell, 0, r2, type=Segment.PROXIMAL)
                    sources = random.sample(np.where(encoded_input)[0], 2)
                    for source in sources:
                        proximal.add_synapse(source, permanence=0.3)
                    cell.proximal_segments = [proximal]
                    cell.excitatory = excitatory
                    cell.fade_rate = FAST_FADE_RATE
                    suffix = 'exc' if excitatory else 'inh'
                    key = 'periph_' + suffix
                    mapping[char][key].append(cell.index)

        # Now we create connections in output region to light up one cell per sequence
        OUTPUT_CONNECTIONS = 6

        for i, seq in enumerate(self.SEQUENCES):
            r3 = self.b.regions[1]
            cell = r3.cells[i]
            # Loop through sequence start to finish
            proximal = Segment(cell, cell.index, r3, type=Segment.PROXIMAL)
            for j, char in enumerate(seq):
                # Earliest chars have the most inhibitory connections
                percent_periph_inh = (len(seq) - j) / float(len(seq))
                n_periph = OUTPUT_CONNECTIONS
                n_periph_inh = int(n_periph * percent_periph_inh)
                char_mapping = mapping[char]
                sources = []
                sources.extend(random.sample(char_mapping['core'], OUTPUT_CONNECTIONS))
                sources.extend(random.sample(char_mapping['periph_exc'], min(n_periph - n_periph_inh, len(char_mapping['periph_exc']))))
                sources.extend(random.sample(char_mapping['periph_inh'], min(n_periph_inh, len(char_mapping['periph_inh']))))
                for s in set(sources):
                    proximal.add_synapse(s, permanence=0.3)
            cell.proximal_segments = [proximal]

    def encode_letter(self, c):
        i = ord(c.upper()) - 65 # A == 0
        return self.encoder.encode(i)

    def run(self):
        self.printer.startloop(self.delay, self.process)

    def start_batch(self, steps=10):
        if steps == 0:
            print("Running to end")
            self.current_batch_target = self.crop_file - self.file_processer.cursor
        else:
            print("Running %d step(s)" % steps)
            self.current_batch_target = steps
        self.current_batch_counter = 0
        self.process()

    def process(self):
        # Process one step
        in_batch = self.current_batch_counter < self.current_batch_target
        if in_batch:
            # Process one step
            self.current_batch_counter += 1
            prior_char = self.char
            self.char, self.sequence_index = self.file_processer.read_next()
            inputs = self.encode_letter(self.char)
            self.printer.set_raw_input(self.char)
            self.b.process(inputs, learning=False)
            active_outputs = [str(x) for x in np.where(self.b.regions[1].activation[:3] >= 1)[0]]
            for i, output in enumerate(active_outputs):
                if i not in self.seq_matches:
                    self.seq_matches[i] = {'correct': 0, 'incorrect': 0}
                correct = bool(output) == (self.sequence_index == i)
                incr_key = 'correct' if correct else 'incorrect'
                self.seq_matches[i][incr_key] += 1
            print "In ground truth sequence %d, active outputs: %s" % (self.sequence_index, ', '.join(active_outputs))
            if self.classifier:
                self.classifier.read(self.char, prior_input=prior_char)
                prediction = self.classifier.predict()
                self.printer.set_prediction(prediction)
            if self.animate:
                self.printer.render()
            self.printer.after(self.delay, self.process)
        else:
            # Wait for user input via tkinter controller window
            pass

    def quit(self):
        print("Quitting...")
        for seq_index, counts in self.seq_matches.items():
            correct = counts.get('correct')
            total = correct + counts.get('incorrect')
            hitrate = 100. * correct / float(total)
            print "Hit rate for sequence %d: %.1f%%" % (seq_index, hitrate)
        self.quitting = True
        self.printer.destroy()


def main(argv):
    processor = DesignedUserInputRunner(animate=True)
    processor.run()


if __name__ == "__main__":
    main(sys.argv[1:])