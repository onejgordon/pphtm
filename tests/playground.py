#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np

from nupic.encoders.scalar import ScalarEncoder

def main():
	encoder = ScalarEncoder(n=5**2, w=5, minval=0, maxval=3, periodic=False, forced=True)
	char = ord("D".upper()) - 65 # A == 0
	print encoder.encode(char)



if __name__ == "__main__":
    main()