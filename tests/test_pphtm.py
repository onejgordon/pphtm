#!/usr/bin/env python
import sys, getopt
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pphtm.pphtm_brain import PPHTMBrain
from chtm.chtm_printer import CHTMPrinter
from pphtm.pphtm_predictor import PPHTMPredictor
from helpers.file_processer import FileProcesser
import numpy as np

from nupic.encoders.scalar import ScalarEncoder
from encoders import SimpleFullWidthEncoder

USE_SIMPLE_ENCODER = True
SHOW_RUN_SUMMARY = False

class TestRunner(object):
    '''
    Runs a single data file through a PPHTM and visualizes processing.
    '''

    def __init__(self, filename=None, with_classifier=True, animate=True, crop=200):
        self.file_processer = FileProcesser(filename=filename)
        self.cats, self.data, self.n_inputs = self.file_processer.open_file()
        self.b = PPHTMBrain(min_overlap=1, r1_inputs=self.n_inputs)
        self.b.initialize()
        self.classifier = None
        self.animate = animate
        self.current_batch_target = 0
        self.current_batch_counter = 0
        self.delay = 50
        self.crop_file = crop
        self.char = None
        self.quitting = False
        if with_classifier:
            self.classifier = PPHTMPredictor(self.b, categories=self.cats)
            self.classifier.initialize()

        if USE_SIMPLE_ENCODER:
            self.encoder = SimpleFullWidthEncoder(n_inputs=self.n_inputs, n_cats=len(self.cats))
        else:
            self.encoder = ScalarEncoder(n=self.n_inputs, w=5, minval=1, maxval=self.n_inputs, periodic=False, forced=True)

        self.printer = CHTMPrinter(self.b, predictor=self.classifier, handle_run_batch=self.start_batch, handle_quit=self.quit)
        self.printer.setup()

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
        finished = self.file_processer.cursor >= self.crop_file
        if finished:
            self.do_prediction()
        else:
            in_batch = self.current_batch_counter < self.current_batch_target
            if in_batch:
                # Process one step
                self.current_batch_counter += 1
                prior_char = self.char
                self.char = self.file_processer.read_next()
                inputs = self.encode_letter(self.char)
                self.printer.set_raw_input(self.char)
                self.b.process(inputs, learning=True)
                if self.classifier:
                    self.classifier.read(self.char, prior_input=prior_char)
                    prediction = self.classifier.predict()
                    self.printer.set_prediction(prediction)
                if self.animate:
                    self.printer.render()
                batch_finished = self.current_batch_counter == self.current_batch_target
                if batch_finished:
                    if SHOW_RUN_SUMMARY and self.current_batch_target != 1:
                        self.printer.show_run_summary()
                self.printer.after(self.delay, self.process)
            else:
                # Wait for user input via tkinter controller window
                pass


    def quit(self):
        print("Quitting...")
        self.printer.destroy()


    def do_prediction(self):
        if self.classifier:
            done = False
            while not done:
                next = raw_input("Enter next letter (q to exit) >> ")
                if next:
                    done = next.upper() == 'Q'
                    if done:
                        break
                    inputs = self.encode_letter(next)
                    self.printer.set_raw_input(next)
                    self.b.process(inputs, learning=False)
                    prediction = self.classifier.predict()
                    self.printer.set_prediction(prediction)
                    if self.animate:
                        self.printer.render()

        self.printer.destroy()


def main(argv):
    HELP = 'test_pphtm.py -c <crop> -f <file>'
    try:
        opts, args = getopt.getopt(argv,"hi:c:",["iterations=","crop="])
    except getopt.GetoptError:
        print HELP
        sys.exit(2)
    # Defaults
    kwargs = {
        'crop': 400,
        'filename': "longer_char_sequences1.txt" # "simple_pattern2.txt"
    }
    for opt, arg in opts:
        if opt == '-h':
            print HELP
            sys.exit()
        elif opt in ("-c", "--crop"):
            kwargs['crop'] = int(arg)
        elif opt in ("-f", "--file"):
            kwargs['filename'] = arg
    processor = TestRunner(animate=True, with_classifier=True, **kwargs)
    processor.run()

if __name__ == "__main__":
   main(sys.argv[1:])