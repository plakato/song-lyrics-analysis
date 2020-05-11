import csv
import os
from os.path import isfile, join

from PIL import Image

import numpy as np

# PARAMETERS
with_line_separators = False


def generate_repetition_matrix(path, filename):
    input_file = path + filename
    song_name = filename.split('\'')[1]
    # Get full lyrics represented as a list of phonemes.
    phon_lyrics = []
    with open(input_file, newline='') as input:
        rows = csv.reader(input, delimiter=';')
        next(rows)
        delimiter = ' '
        for row in rows:
            row_phons = row[3].replace('_', ' ').split(delimiter)
            phon_lyrics += row_phons + ([delimiter] if with_line_separators
                                        else [])

    # Draw the image
    # Create a WxHx3 array of 8 bit unsigned integers
    size = len(phon_lyrics)
    pixels = np.zeros((size, size, 3), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            if phon_lyrics[i] == phon_lyrics[j]:
                pixels[i, j] = [0, 0, 0]
            elif phon_lyrics[i] == delimiter:
                pixels[i, j] = [0, 0, 255]
            else:
                pixels[i, j] = [255, 255, 255]

    # Use PIL to create an image from the new array of pixels
    matrix = Image.fromarray(pixels)
    matrix.save('sparsar_experiments/repetition_matrices/' + (
        'with_linesep_' if with_line_separators else '') + song_name + '.png')
    print('Generated', song_name)


# Generate matrices for all files.
path = 'sparsar_experiments/rhymes/'
for item in os.listdir(path):
    if isfile(join(path, item)) and item.endswith('.csv'):
        generate_repetition_matrix(path, item)
