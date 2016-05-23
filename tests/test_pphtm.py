#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pphtm.pphtm_brain import PPHTMBrain
from chtm.chtm_printer import CHTMPrinter
from pphtm.pphtm_predictor import PPHTMPredictor
import numpy as np

from nupic.encoders.scalar import ScalarEncoder
from encoders import SimpleFullWidthEncoder

USE_SIMPLE_ENCODER = True
FILENAME = "longer_char_sequences1.txt"
SHOW_RUN_SUMMARY = True
# FILENAME = "simple_pattern2.txt"

class FileProcesser(object):

    DATA_DIR = "../data"
    ALPHA = "ABCDEFG"
    CROP_FILE = 300
    N_INPUTS = 7**2

    def __init__(self, filename=FILENAME, with_classifier=True, delay=50, animate=True):
        self.b = PPHTMBrain(min_overlap=1, r1_inputs=self.N_INPUTS)
        self.b.initialize()
        self.classifier = None
        self.animate = animate
        self.current_batch_target = 0
        self.current_batch_counter = 0
        self.delay = delay
        if with_classifier:
            self.classifier = PPHTMPredictor(self.b, categories=self.ALPHA)
            self.classifier.initialize()

        if USE_SIMPLE_ENCODER:
            self.encoder = SimpleFullWidthEncoder(n_inputs=self.N_INPUTS, n_cats=len(self.ALPHA))
        else:
            self.encoder = ScalarEncoder(n=self.N_INPUTS, w=5, minval=1, maxval=self.N_INPUTS, periodic=False, forced=True)

        self.printer = CHTMPrinter(self.b, predictor=self.classifier)
        self.printer.setup()

        with open(self.DATA_DIR + "/" + filename, 'r') as myfile:
            self.data = myfile.read()

        self.cursor = 0

    def encode_letter(self, c):
        i = ord(c.upper()) - 65 # A == 0
        return self.encoder.encode(i)

    def run(self):
        self.printer.window.after(self.delay, self.process)
        self.printer.window.mainloop()

    def process(self):
        finished = self.cursor >= self.CROP_FILE
        if finished:
            self.do_prediction()
        else:
            in_batch = self.current_batch_counter < self.current_batch_target
            if in_batch:
                # Process one step
                self.cursor += 1
                self.current_batch_counter += 1
                char = self.data[self.cursor].upper()

                inputs = self.encode_letter(char)
                self.printer.set_raw_input(char)
                self.b.process(inputs, learning=True)
                if self.classifier:
                    self.classifier.read(char)
                    prediction = self.classifier.predict()
                    self.printer.set_prediction(prediction)
                if self.animate:
                    self.printer.render()
                batch_finished = self.current_batch_counter == self.current_batch_target
                if batch_finished and SHOW_RUN_SUMMARY and self.current_batch_target != 1:
                    self.printer.show_run_summary()
            else:
                # Get user input for next batch
                self.current_batch_counter = 0
                n_steps = raw_input("Enter # of steps to run, or 0 to run to end, 'q' to quit...")
                digit = n_steps.isdigit()
                quit = n_steps.upper() == "Q"
                if quit:
                    self.printer.window.destroy()
                    return
                if not digit:
                    n_steps = 1
                else:
                    n_steps = int(n_steps)
                if n_steps == 0:
                    self.current_batch_target = self.CROP_FILE - self.cursor
                else:
                    self.current_batch_target = n_steps

            self.printer.window.after(self.delay, self.process)

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

        self.printer.window.destroy()

def main():
    processor = FileProcesser(delay=10, animate=True, with_classifier=True)
    processor.run()

if __name__ == "__main__":
    main()
