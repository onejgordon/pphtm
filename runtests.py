#!/usr/bin/python
import optparse
import sys
import os
import unittest

USAGE = """%prog SDK_PATH TEST_PATH
Run unit tests for App Engine apps.

SDK_PATH    Path to the SDK installation
TEST_PATH   Path to package containing test modules
MODULE      Name of test module to run (optional, if not provided, run all)
"""


def main(test_path, module=None):
    p = os.path.abspath(os.path.dirname(test_path))
    sys.path.append(p)

    if module and module != 'all':
        suite = unittest.loader.TestLoader().discover(test_path, pattern=module)
    else:
        suite = unittest.loader.TestLoader().discover(test_path)
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if test_result.wasSuccessful() else 1)


if __name__ == '__main__':
    parser = optparse.OptionParser(USAGE)
    options, args = parser.parse_args()
    if len(args) < 2:
        print 'Error: 2+ arguments required.'
        parser.print_help()
        sys.exit(1)
    TEST_PATH = args[0]
    MODULE = args[1] if len(args) > 2 else None
    main(TEST_PATH, module=MODULE)
