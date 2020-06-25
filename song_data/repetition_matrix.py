import csv
import os
from os.path import isfile, join

from PIL import Image

import numpy as np

# PARAMETERS
with_line_separators = True


def generate_endrhyme_matrix(path, filename, save_path):
    input_file = path + filename
    song_name = filename.replace('.csv', '').replace('.txt', '').replace(
        '\'', '')
    lines = []
    with open(input_file, newline='') as input:
        rows = csv.reader(input, delimiter=';')
        next(rows)
        for row in rows:
            lines.append(row[0])
    return create_and_save_image(lines, None, song_name, save_path)


def generate_phoneme_matrix(path, filename):
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
            phon_lyrics += row_phons[-4:] + (
                [delimiter] if with_line_separators else [])
    save_path = 'sparsar_experiments/repetition_matrices/phonemes' + (
        '_with_linesep/' if with_line_separators else '/')
    return create_and_save_image(phon_lyrics, delimiter, song_name, save_path)


# Returns binary matrix - True represents matching pair.
def create_and_save_image(data, delimiter, song_name, path):
    # Draw the image
    # Create a WxHx3 array of 8 bit unsigned integers
    size = len(data)
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    bin_matrix = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if data[i] == data[j] and data[i] != delimiter:
                bin_matrix[i, j] = True
                pixels[i, j] = [0, 0, 0]
            elif data[i] == delimiter or data[j] == delimiter:
                pixels[i, j] = [200, 200, 255]
            else:
                pixels[i, j] = [255, 255, 255]

    # Use PIL to create an image from the new array of pixels
    matrix = Image.fromarray(pixels)
    matrix.save(path + song_name + '.png')
    print('Generated', song_name, 'matrix.')
    return bin_matrix


if __name__ == '__main__':
    # Generate matrices for all files.
    path = 'sparsar_experiments/rhymes/original/'
    for item in os.listdir(path):
        if isfile(join(path, item)) and item.endswith('.csv'):
            save_path = 'sparsar_experiments/repetition_matrices/endrhymes_original/'
            generate_endrhyme_matrix(path, item, save_path)
            # generate_phoneme_matrix(path, item)
