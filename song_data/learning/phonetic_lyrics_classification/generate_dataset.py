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
phoneme_sep = '/'
line_sep = ';'
stanza_sep = '!'
# Have to use space, because SubtextEncoder counts on it.
word_sep = ' '


def get_data(dir):
    data = []
    for item in os.listdir(dir):
        file_name = join(dir, item)
        if isfile(file_name) and item.endswith('.csv'):
            df = pd.read_csv(file_name, sep=';')
            phonemes_list = df['Phonetic Transcription'].tolist()
            phonemes_str = []
            for line in phonemes_list:
                # Dataset contains NaNs - lines that sparsar wasn't able to parse, not valid lyrics.
                if isinstance(line, str):
                    if line == '':
                        phonemes_str.append(stanza_sep)
                    else:
                        phonemes_str.append(line.replace('_', phoneme_sep))
            phonemes_str = line_sep.join(phonemes_str)
            data.append(phonemes_str)
    return data


def print_data_statistics(data):
    # Initialization of variables.
    # For calculating average, min, max # of phonemes per song:
    min_phon = 10000
    max_phon = 0
    sum_phonemes = 0
    total_examples = 0
    # Different unique phonemes
    phonemes_unique = set()

    # Calculation of the statistics.
    for song in data:
        total_examples += 1
        phonemes = re.split(line_sep + '|' +
                            word_sep + '|' +
                            phoneme_sep + '|' +
                            stanza_sep, song)
        for phoneme in phonemes:
            phonemes_unique.add(phoneme)
        length = len(phonemes)
        if length < min_phon:
            min_phon = length
        if length > max_phon:
            max_phon = length
        sum_phonemes += length

    # Printing the results.
    print('Song length in phonemes')
    print('Min:', min_phon)
    print('Max:', max_phon)
    print('Avg:', sum_phonemes/total_examples)
    print('Total examples:', total_examples)
    print('Unique phonemes:', len(phonemes_unique))
    print(phonemes_unique)


def encode_subwords(data, vocab_filename='vocab'):
    gen = (song for song in data)
    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(gen, target_vocab_size=2 ** 15)
    encoder.save_to_file(vocab_filename)


def prepare_data_for_learning(true_data, test_split=0.3):
    # Generate false examples by shuffling lines.
    false_data = []
    for example in true_data:
        example = example.split(line_sep)
        false_ex = random.sample(example, len(example))
        false_ex = line_sep.join(false_ex)
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
    with open('dataset/train/shuffled.txt', 'w+') as train_shuf:
        for i in range(0, split_idx):
            train_shuf.write(false_data[i] + '\n')
    with open('dataset/val/original.txt', 'w+') as val_orig:
        for i in range(split_idx, len(true_data)):
            val_orig.write(true_data[i] + '\n')
    with open('dataset/val/shuffled.txt', 'w+') as val_shuf:
        for i in range(split_idx, len(false_data)):
            val_shuf.write(false_data[i] + '\n')


if __name__ == '__main__':
    dir='../../sparsar_experiments/rhymes/original'
    data = get_data(dir)
    print_data_statistics(data)
    encode_subwords(data)
    prepare_data_for_learning(data)
