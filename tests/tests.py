#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from htm_brain import HTMBrain
import numpy as np

INPUT_LEN = 5
DURATION = 1
LEARNING = True
b = HTMBrain(columns_per_region=[30,10,5])            
# Run white noise
b.initialize(r1_inputs=INPUT_LEN)
for i in range(DURATION):
    out = b.process(np.random.choice(2, INPUT_LEN), learning=LEARNING)
