#!/usr/bin/env python

import sys

class FileProcesser(object):
    '''
    Helper for loading data files for processing
    Assumes standard format with full cat list on 1st line
    '''

    DATA_DIR = "../data"

    def __init__(self, filename=None, with_sequences=False):
        self.filename = filename
        self.data = None
        self.cats = None
        self.n_inputs = 0
        self.cursor = 0
        self.with_sequences = with_sequences

    def open_file(self):
        with open(self.DATA_DIR + "/" + self.filename, 'r') as myfile:
            self.cats = myfile.readline().replace('\n','')
            self.data = myfile.readline()
            if self.with_sequences:
                self.sequences = myfile.readline().replace('\n','')
            self.n_inputs = len(self.cats) ** 2
        print "Loaded %s (len %d, %d cats: %s)" % (self.filename, len(self.data), len(self.cats), self.cats)
        return (self.cats, self.data, self.n_inputs)

    def read_next(self):
        # Process one step
        self.cursor += 1
        char = self.data[self.cursor].upper()
        if self.with_sequences:
            sequence = self.sequences[self.cursor]
            if sequence == "-":
                sequence = None
            else:
                sequence = int(sequence)
            return (char, sequence)
        else:
            return char

    def reset(self):
        self.cursor = 0
