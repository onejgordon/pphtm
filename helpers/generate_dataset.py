#!/usr/bin/env python
import sys
from os import path
import random
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

DIR = "../data/"
FILENAME = "longer_char_sequences1.txt"

CHARS = "ABCDEFG"

WORDS = [
	"ABCDEFG",
	"BAGGD",
	"GAGGF",
	"GAFFED",
	"BAD"
]

CHANCE_NOISE = 0.01
CHANCE_CORRUPT_WORD = 0.01
NOISE_LEN_RANGE = (1,5)
FILE_LEN = 500

def random_char():
	index = random.randint(0, len(CHARS)-1)
	return CHARS[index]

data = ""
while len(data) < FILE_LEN:
	noise = random.random() <= CHANCE_NOISE
	if noise:
		noise_len = random.randint(NOISE_LEN_RANGE[0], NOISE_LEN_RANGE[1])
		for i in range(noise_len):
			data += random_char()
	else:
		corrupt_word = random.random() <= CHANCE_CORRUPT_WORD
		word = random.choice(WORDS)
		if corrupt_word:
			# Replace 1 char with a random char
			word = list(word)
			index = random.randint(0, len(word)-1)
			word[index] = random_char()
			word = ''.join(word)
		data += word

print "Writing %d characters to %s" % (len(data), DIR+FILENAME)

with open(DIR+FILENAME, "w") as text_file:
	text_file.write(data)
