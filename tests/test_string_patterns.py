#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from htm_brain import HTMBrain
import numpy as np

b = HTMBrain(columns_per_region=[60,10], min_overlap=1) 
b.initialize(r1_inputs=26)

data = open("string_data.txt", 'r')

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CROP_FILE = 10

def encode_letter(c):
    index = ALPHA.find(c.upper())
    input = [0 for x in range(26)]
    input[index] = 1
    return input

def read():
    print "Learning..."
    i = 0
    done = False
    for line in data:
        if not done:
            for letter in line:
                if i < CROP_FILE or not CROP_FILE:
                    print "Processing %s" % letter
                    out = b.process(encode_letter(letter), learning=True)
                    i += 1
                else:
                    return

read()

print "Enter next letter..."
letter = raw_input("> ")
out = b.process(encode_letter(letter))
# How to get a prediction from temporal pooler
