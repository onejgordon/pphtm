#!/usr/bin/python
# -*- coding: utf8 -*-

from datetime import datetime, timedelta
import util
import json
import math
import logging
import os
import unittest
import numpy as np

class UtilTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def testOverlap(self):
        volley = [
            (np.array([1,0,0,5]), np.array([1,0,0,5]), 4),
            (np.array([1,0,0,5]), np.array([1,0,0,5]), 4),
            (np.array([1,0,0,5]), np.array([1,0,1,5]), 3),
            (np.array([1,0,0,0]), np.array([0,1,1,5]), 0),
            (np.array([1,0,0,0,0,0,0]), np.array([0,1,1,5,0,0,0]), 3)
        ]

        for v in volley:
            arr1, arr2, n_overlap = v
            _n_overlap = util.bool_overlap(arr1, arr2)
            self.assertEqual(_n_overlap, n_overlap)


    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()