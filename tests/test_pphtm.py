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

class FileProcesser(object):

    DATA_DIR = "../data"
    ALPHA = "ABCDEF"
    CROP_FILE = 200
    N_INPUTS = 36
    auto_predict = 16

    def __init__(self, filename="simple_pattern2.txt", with_classifier=True, delay=50, animate=True):
        self.b = PPHTMBrain(min_overlap=1, r1_inputs=self.N_INPUTS)
        self.b.initialize(CELLS_PER_REGION=9**2, N_REGIONS=1)
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
            if self.auto_predict:
                predicted_stream = ""
                for i in range(self.auto_predict):
                    # Loop through predicting, and then processing prediction
                    prediction = self.classifier.predict()
                    predicted_stream += prediction
                    inputs = self.encode_letter(prediction)
                    self.b.process(inputs, learning=False)
                print "Predicted stream:", predicted_stream
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

            else:
                while True:
                    user_char = raw_input("Enter a letter to see prediction at t+1... (! to exit) >> ")
                    if user_char == "!":
                        break
                    else:
                        inputs = self.encode_letter(user_char)
                        self.b.process(inputs, learning=False)
                        prediction = self.classifier.predict()

        self.printer.window.destroy()

def main():
    processor = FileProcesser(delay=10, animate=True, with_classifier=True)
    processor.run()

if __name__ == "__main__":
    main()
