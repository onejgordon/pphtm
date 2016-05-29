#!/usr/bin/env python

class FileProcesser(object):
    '''
    Helper for loading data files for processing
    Assumes standard format with full cat list on 1st line
    '''

    DATA_DIR = "../data"

    def __init__(self, filename=None):
        self.filename = filename
        self.data = None
        self.cats = None
        self.n_inputs = 0
        self.cursor = 0

    def open_file(self):
        with open(self.DATA_DIR + "/" + self.filename, 'r') as myfile:
            self.cats = myfile.readline().replace('\n','')
            self.data = myfile.readline()
            self.n_inputs = len(self.cats) ** 2
        print "Loaded %s (len %d, %d cats: %s)" % (self.filename, len(self.data), len(self.cats), self.cats)
        return (self.cats, self.data, self.n_inputs)

    def read_next(self):
        # Process one step
        self.cursor += 1
        return self.data[self.cursor].upper()

    def reset(self):
        self.cursor = 0
