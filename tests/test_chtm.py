#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from chtm_brain import CHTMBrain
from chtm_printer import CHTMPrinter
from chtm_classifier import CHTMClassifier
import numpy as np

from nupic.encoders.scalar import ScalarEncoder

# Notes:
# - We appear not to be learning. Ave # of connected synapses 
# stays constant for each region after a small T.

class FileProcesser(object):

    ALPHA = "ABCDEF"
    CROP_FILE = 200
    N_INPUTS = 36
    AUTO_PREDICT = 0
    CPR = [11**2]

    def __init__(self, filename="simple_pattern2.txt", step=False, with_classifier=True, delay=50, animate=True):
        self.step = step
        self.b = CHTMBrain(cells_per_region=self.CPR, min_overlap=1, r1_inputs=self.N_INPUTS) 
        self.b.initialize()
        self.printer = CHTMPrinter(self.b)
        self.printer.setup()
        self.classifier = None
        self.animate = animate
        self.delay = delay
        if with_classifier:
            self.classifier = CHTMClassifier(self.b, categories=self.ALPHA, region_index=len(self.CPR)-1, history_window=500)

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

    def do_prediction(self):
        if self.classifier and self.AUTO_PREDICT:
            predicted_stream = ""
            for x in range(self.AUTO_PREDICT):
                # Run prediction back in
                prediction = self.classifier.predict(k=1)
                predicted_stream += prediction
                inputs = self.encode_letter(prediction)
                self.b.process(inputs, learning=False)
            print "Predicted stream: %s" % predicted_stream
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


    def process(self):
        if self.cursor < self.CROP_FILE or not self.CROP_FILE:
            if self.step:
                user_val = raw_input("Press enter to step, anything else to quit...")
                if user_val:
                    self.printer.window.destroy()
                    return
            self.cursor += 1
            char = self.data[self.cursor].upper()
            inputs = self.encode_letter(char)
            self.b.process(inputs, learning=True)
            if self.classifier:
                self.classifier.read(char)

            if self.animate:
                self.printer.render(inputs=inputs)

            done = False              
            self.printer.window.after(self.delay, self.process)
        else:
            done = True
        if done:
            self.do_prediction()

                

def main():
    processor = FileProcesser(step=False, delay=10, animate=True)
    processor.run()

if __name__ == "__main__":
    main()
