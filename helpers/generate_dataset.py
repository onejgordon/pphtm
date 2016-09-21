#!/usr/bin/env python
import sys
from os import path
import random
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

DIR = "../data/"
FILENAME = "spatial_sequences.txt"

CHARS = "ABCD"

WORDS = [
    "BACD",
    "ABCD",
    "DAAC"
]

CHANCE_NOISE = 0.0
CHANCE_CORRUPT_WORD = 0.0
NOISE_LEN_RANGE = (1,5)
FILE_LEN = 1000
RECOGNITION_DELAY = 1.5
DRY_RUN = False
SEQUENCE_MARKERS = True  # Add a line identifying which sequence we're in at each step

def best_correct_rate():
    first_letter_errors = len(WORDS) * RECOGNITION_DELAY
    total_letters = sum([len(word) for word in WORDS])
    max_correct = float(total_letters - first_letter_errors)
    return max_correct / total_letters

def random_char():
    index = random.randint(0, len(CHARS)-1)
    return CHARS[index]

data = ""
sequence_markers = ""

while len(data) < FILE_LEN:
    noise = random.random() <= CHANCE_NOISE
    if noise:
        noise_len = random.randint(NOISE_LEN_RANGE[0], NOISE_LEN_RANGE[1])
        for i in range(noise_len):
            data += random_char()
        sequence_markers += "-"
    else:
        corrupt_word = random.random() <= CHANCE_CORRUPT_WORD
        word = random.choice(WORDS)
        seq_index = WORDS.index(word)
        if corrupt_word:
            # Replace 1 char with a random char
            word = list(word)
            index = random.randint(0, len(word)-1)
            word[index] = random_char()
            word = ''.join(word)
        data += word
        sequence_markers += str(seq_index) * len(word) # Even if corrupt

print "Writing %d characters to %s" % (len(data), DIR+FILENAME)
print "Best correct rate: %.1f" % (best_correct_rate() * 100.)

# Writes all cats (chars) to first line, and full data block to second
if not DRY_RUN:
    with open(DIR+FILENAME, "w") as text_file:
        text_file.write(CHARS+'\n')
        text_file.write(data)
        if SEQUENCE_MARKERS:
            text_file.write('\n'+sequence_markers)
