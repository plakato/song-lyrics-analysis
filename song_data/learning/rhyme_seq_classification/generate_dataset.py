import os
import random
from os.path import isfile, join
import pandas as pd
import numpy as np
from pathlib import Path


def get_data(dir):
    data = []
    for item in os.listdir(dir):
        file_name = join(dir, item)
        if isfile(file_name) and item.endswith('.csv'):
            df = pd.read_csv(file_name, sep=';')
            letters = df['Rhyme Scheme Letter'].tolist()
            data.append(letters)
    return data


def create_vocabulary(data, file):
    # Isolate unique.
    unique = set()
    for letters in data:
        for letter in letters:
            unique.add(letter)
    # Save to vocab file.
    with open(file, 'w+') as vocab_file:
        for value in unique:
            vocab_file.write(value + '\n')

def print_data_statistics(data):
    # Initialization of variables.
    # Information concerning number of different rhyme classes.
    min_classes = 1000
    max_classes = 0
    # For calculating average # of classes:
    sum_no_classes = 0
    total_examples = 0
    # Information concerning length - number of verses.
    min_len = 1000
    max_len = 0
    sum_len = 0

    # Calculation of the statistics.
    for letters in data:
        total_examples += 1
        length = len(letters)
        if length < min_len:
            min_len = length
        if length > max_len:
            max_len = length
        sum_len += length
        unique = len(np.unique(letters))
        if unique < min_classes:
            min_classes = unique
        if unique > max_classes:
            max_classes = unique
        sum_no_classes += unique

    # Printing the results.
    print('Unique rhyme classes in one example:')
    print('Min:', min_classes)
    print('Max:', max_classes)
    print('Avg:', sum_no_classes/total_examples)
    print('Length of one example - number of verses:')
    print('Min:', min_len)
    print('Max:', max_len)
    print('Avg:', sum_len/total_examples)


def prepare_data_for_learning(true_data, test_split=0.3, separator=' '):
    # Generate false examples.
    false_data = []
    for example in true_data:
        false_ex = random.sample(example, len(example))
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
            train_orig.write(separator.join(true_data[i]) + '\n')
    with open('dataset/train/shuffled.txt', 'w+') as train_shuf:
        for i in range(0, split_idx):
            train_shuf.write(separator.join(false_data[i]) + '\n')
    with open('dataset/val/original.txt', 'w+') as val_orig:
        for i in range(split_idx, len(true_data)):
            val_orig.write(separator.join(true_data[i]) + '\n')
    with open('dataset/val/shuffled.txt', 'w+') as val_shuf:
        for i in range(split_idx, len(false_data)):
            val_shuf.write(separator.join(false_data[i]) + '\n')


if __name__ == '__main__':
    dir='../../sparsar_experiments/rhymes/original'
    data = get_data(dir)
    print_data_statistics(data)
    create_vocabulary(data, 'dataset/vocab.txt')
    prepare_data_for_learning(data)
