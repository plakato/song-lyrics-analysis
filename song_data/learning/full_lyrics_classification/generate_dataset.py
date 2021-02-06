import json
import os
import random
import re
from os.path import isfile, join
import tensorflow_datasets as tfds

import math
import pandas as pd
import numpy as np
from pathlib import Path

# CONSTANTS
# Can't use "_" -> special char in SubtextEncoder, nor "\n" -> used in saving dataset.
line_end = ';'
stanza_end = '#'
song_start = '>'
# Maximum number of characters per song. Exclude too long inputs to speed up learning.
max_char_count = 30*1000


def get_data(filename):
    data = []
    with open(filename) as input:
        songs = json.load(input)
    for song in songs:
        lines = song['lyrics']
        lyrics = song_start
        for line in lines:
            # Make lowercase for BERT.
            line = line.lower()
            # Add markers.
            if line == '' and lyrics[-1] != stanza_end and lyrics[-1] != song_start:
                lyrics += stanza_end
            if lyrics[-1] == stanza_end or lyrics[-1] == song_start:
                lyrics += line.strip()
            else:
                lyrics += line_end + line.strip()
        if lyrics[-1] != stanza_end:
            lyrics += stanza_end
        if len(lyrics) < max_char_count:
            data.append(lyrics)
    return data


def print_data_statistics(data):
    print('Total songs:', len(data))


def encode_subwords(data, vocab_filename='vocab'):
    gen = (song for song in data)
    encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(gen, target_vocab_size=2 ** 15)
    encoder.save_to_file(vocab_filename)
    print('SubwordTextEncoder vocabulary size is', encoder.vocab_size)


def prepare_data_for_learning(true_data, test_split=0.3):
    # Generate false examples by shuffling lines.
    false_data = []
    for example in true_data:
        verses = re.split(line_end+'|'+stanza_end, example)
        verses = verses[:-1]
        false_verses = random.sample(verses, len(verses))
        special_chars = re.findall('['+stanza_end + '|' + line_end + '|' + song_start + ']', example)
        false_ex = ''
        for line in false_verses:
            false_ex += special_chars.pop(0) + line
        false_ex += special_chars.pop()
        false_data.append(false_ex)
    # Shuffle.
    random.shuffle(true_data)
    random.shuffle(false_data)
    # Save.
    split_idx = int(len(true_data)*(1-test_split))
    Path("dataset/train").mkdir(parents=True, exist_ok=True)
    Path("dataset/val").mkdir(parents=True, exist_ok=True)
    with open('dataset/train/original.txt', 'w+') as train_orig:
        for i in range(0, split_idx):
            train_orig.write(true_data[i] + '\n')
    print('Generated {0} true examples for training.'.format(split_idx))
    with open('dataset/train/shuffled.txt', 'w+') as train_shuf:
        for i in range(0, split_idx):
            train_shuf.write(false_data[i] + '\n')
    print('Generated {0} false examples for training.'.format(split_idx))
    with open('dataset/val/original.txt', 'w+') as val_orig:
        for i in range(split_idx, len(true_data)):
            val_orig.write(true_data[i] + '\n')
    print('Generated {0} true examples for validation.'.format(len(true_data) - split_idx))
    with open('dataset/val/shuffled.txt', 'w+') as val_shuf:
        for i in range(split_idx, len(false_data)):
            val_shuf.write(false_data[i] + '\n')
    print('Generated {0} false examples for validation.'.format(len(false_data) - split_idx))


if __name__ == '__main__':
    file = '../../data/ENlyrics_cleaned.json'
    data = get_data(file)
    print_data_statistics(data)
    # encode_subwords(data)
    prepare_data_for_learning(data)
