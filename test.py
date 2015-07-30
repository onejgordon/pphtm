#!/usr/bin/env python

import numpy as np
import util

v = [0, 1, 0]
axis = [0, 0, 1]
theta = np.pi / 6

for x in range(15):
	v = np.dot(util.rotation_matrix(axis,theta), v)
	bearing = util.bearingFromVector(v[:2])
	print bearing
	