#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import util

def testUtils():
	x, y = util.coords_from_index(3, 10)
	assert x == 3
	assert y == 0

	x, y = util.coords_from_index(14, 10)
	assert x == 4
	assert y == 1

	x, y = util.coords_from_index(0, 10)
	assert x == 0
	assert y == 0

	history = []
	util.rolling_list(history, length=3, new_value=1)
	assert len(history) == 1

	util.rolling_list(history, length=3, new_value=2.5)
	assert sum(history) == 3.5


def main():
	testUtils()


if __name__ == "__main__":
    main()

