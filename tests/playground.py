#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np

from nupic.encoders.scalar import ScalarEncoder

def main():
	# encoder = ScalarEncoder(n=5**2, w=5, minval=0, maxval=10, periodic=False, forced=True)

	# volley = range(1,10)
	# for v in volley:
	# 	print encoder.encode(v)

	new_list = ['130 kms', '46 kms', '169 kms', '179 kms', '53 kms', '128 kms', '97 kms', '152 kms', '20 kms', '94 kms', '$266.07', '$174.14', '$187.01', '$488.69', '$401.53', '$106.88', '$398.33', '$493.87', '$205.43', '$248.14']
	n_items = len(new_list) / 2
	# sorted_list = [[new_list[i], new_list[i+n_items]] for i in range(n_items)]

	distances = new_list[:n_items]
	prices = new_list[n_items:]
	sorted_list = zip(distances, prices)




if __name__ == "__main__":
    main()