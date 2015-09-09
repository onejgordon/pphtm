#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from htm_brain import HTMBrain
import numpy as np

INPUT_LEN = 10
DURATION = 10
LEARNING = True
b = HTMBrain()
# Run white noise
b.initialize(n_regions=2, input_length=INPUT_LEN)
for i in range(DURATION):
    out = b.process(np.random.choice(2, INPUT_LEN), learning=LEARNING)
