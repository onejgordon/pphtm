#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from chtm_brain import CHTMBrain
from chtm_printer import CHTMPrinter
from chtm_classifier import CHTMClassifier
import numpy as np

from nupic.encoders.scalar import ScalarEncoder

class FileProcesser(object):

    ALPHA = "ABCDEF"
    CROP_FILE = 500
    N_INPUTS = 36
    CPR = [8**2]
    auto_predict = 16

    def __init__(self, filename="simple_pattern.txt", with_classifier=True, delay=50, animate=True):
        self.b = CHTMBrain(cells_per_region=self.CPR, min_overlap=1, r1_inputs=self.N_INPUTS) 
        self.b.initialize()
        self.printer = CHTMPrinter(self.b)
        self.printer.setup()
        self.classifier = None
        self.animate = animate
        self.current_batch_target = 0
        self.current_batch_counter = 0
        self.delay = delay
        if with_classifier:
            self.classifier = CHTMClassifier(self.b, categories=self.ALPHA, region_index=len(self.CPR)-1, history_window=self.CROP_FILE/2)

        self.encoder = ScalarEncoder(n=self.N_INPUTS, w=5, minval=1, maxval=self.N_INPUTS, periodic=False, forced=True)

        with open(filename, 'r') as myfile:
            self.data = myfile.read()

        self.cursor = 0    

    def encode_letter(self, c):
        if True:
            # Manual encoding (simple)
            i = ord(c) - 65 # A = 0
            n_cats = len(self.ALPHA)
            offset = n_cats*i
            inputs = np.zeros(self.N_INPUTS)
            inputs[offset:offset+n_cats] = 1
            return inputs
        else:
            i = ord(c) - 64
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
                self.b.process(inputs, learning=True)
                if self.classifier:
                    self.classifier.read(char)

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
                    prediction = self.classifier.predict(k=1)
                    predicted_stream += prediction
                    inputs = self.encode_letter(prediction)
                    self.b.process(inputs, learning=False)
                print "Predicted: %s" % predicted_stream
                done = False
                while not done:
                    next = raw_input("Enter next letter (q to exit) >> ")
                    done = next.upper() == 'Q'
                    if done:
                        break
                    inputs = self.encode_letter(next)
                    self.b.process(inputs, learning=False)
                    prediction = self.classifier.predict(k=1)
                    print "Prediction: %s" % prediction
            else:
                while True:            
                    user_char = raw_input("Enter a letter to see prediction at t+1... (! to exit) >> ")
                    if user_char == "!":
                        break
                    else:
                        inputs = self.encode_letter(user_char)
                        self.b.process(inputs, learning=False)
                        prediction = self.classifier.predict(k=1)
                        print "Prediction: %s" % prediction

        self.printer.window.destroy()

def main():
    processor = FileProcesser(delay=10, animate=True, with_classifier=True)
    processor.run()

if __name__ == "__main__":
    main()
