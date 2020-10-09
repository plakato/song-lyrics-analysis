import itertools
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
# Epoch separator.
epoch_sep = 'END OF EPOCH\n'


# Return given amount `count` of rhyming verse pairs and the same amount of not rhyming verse pairs from each song.
def get_data(dir, count=10):
    data_rhyming = []
    data_not_rhyming = []
    for item in os.listdir(dir):
        file_name = join(dir, item)
        if isfile(file_name) and item.endswith('.csv'):
            df = pd.read_csv(file_name, sep=';')
            lyrics = df['Lyrics'].tolist()
            rhyme_classes = df['Rhyme Scheme Letter'].tolist()
            not_rhyming = get_verse_pairs(lyrics, rhyme_classes, count, False)
            # Don't use the songs where everything rhymes with everything.
            if not_rhyming is None:
                continue
            rhyming = get_verse_pairs(lyrics, rhyme_classes, count, True)
            data_rhyming.append(rhyming)
            data_not_rhyming.append(not_rhyming)
    return data_rhyming, data_not_rhyming


def get_verse_pairs(lyrics, rhyme_classes, count, rhyming):
    not_found = 'NOT FOUND'
    pairs = []
    for i in range(count):
        # Generate two different random indexes.
        n = random.randint(0, len(lyrics) - 1)
        m = random.randint(0, len(lyrics) - 1)
        start_n = n
        while m == n:
            m = random.randint(0, len(lyrics) - 1)
        # Move to the next line if lines don't follow rhyming/not_rhyming parameter.
        iteration = 0
        # Try all possible m values, then try to change n.
        while ((rhyming and (rhyme_classes[n] != rhyme_classes[m] or n == m)) or
               (not rhyming and (rhyme_classes[n] == rhyme_classes[m] or n == m))):
            m = (m + 1) % len(lyrics)
            iteration += 1
            if iteration >= len(lyrics):
                n = (n + 1) % len(lyrics)
                iteration = 0
                if n == start_n:
                    print('Data not generated for following lyrics:', lyrics)
                    return None
        pairs.append([lyrics[n], lyrics[m]])
    return pairs


def encode_subwords(data, vocab_filename='vocab'):
    # Convert each array to one string.
    gen = (' '.join([item for touple in song for item in touple]) for song in data)
    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(gen, target_vocab_size=2 ** 15)
    encoder.save_to_file(vocab_filename)


def prepare_data_for_learning(data, label, test=3):
    total_count = len(data)
    song_count = len(data[0])
    # Reorganize to sets. Each set contains only one verse couple from each song. Shuffle.
    data_shuffled = []
    for j in range(song_count):
        column = []
        for i in range(total_count):
            column.append(data[i][j])
        random.shuffle(column)
        data_shuffled.append(column)
    # Save.
    train = song_count - test
    Path("dataset/train").mkdir(parents=True, exist_ok=True)
    Path("dataset/val").mkdir(parents=True, exist_ok=True)
    # Validation doesn't have epochs so let's flatten the data as one epoch.
    val_data = [list(itertools.chain.from_iterable(data_shuffled[train:]))]
    write_to_file(data_shuffled[:train], 'dataset/train/' + label + '.txt')
    write_to_file(val_data, 'dataset/val/' + label + '.txt')


def write_to_file(data, filename):
    with open(filename, 'w+') as output:
        for epoch in range(len(data)):
            for touple in range(len(data[epoch])):
                output.write(data[epoch][touple][0] + '\n')
                output.write(data[epoch][touple][1] + '\n')
            output.write(epoch_sep)


if __name__ == '__main__':
    # pairs = get_verse_pairs(['nie','ci','ano', 'vrie','ti', 'voda'],['a', 'b', 'c', 'a', 'b','d'], 2, True)
    dir='../../sparsar_experiments/rhymes/original'
    count = 10
    rhyming, not_rhyming = get_data(dir, count)
    encode_subwords(rhyming+not_rhyming)
    prepare_data_for_learning(rhyming, label='rhyming', test=3)
    prepare_data_for_learning(not_rhyming, label='not_rhyming', test=3)