import os
from os.path import isfile, join
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter


class RhymeSeqBatchGenerator(tf.keras.utils.Sequence):

    def __init__(self, BATCH_SIZE=1, dir='../../sparsar_experiments/rhymes/original'):
        """ Initialize Generator object.
        Args
            DATASET_PATH           : Path to folder containing individual folders named by their class names
            BATCH_SIZE             : The size of the batches to generate.
            shuffle_images         : If True, shuffles the images read from the DATASET_PATH
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
        """

        self.batch_size = BATCH_SIZE
        data = self.get_data(dir)
        self.print_data_statistics(data)
        self.prepare_data_for_learning(data)

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.seq_groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        seq_group = self.seq_groups[index]
        label_group = self.label_groups[index]

        return np.array(seq_group), np.array(label_group)

    def get_data(self, dir):
        data = []
        for item in os.listdir(dir):
            file_name = join(dir, item)
            if isfile(file_name) and item.endswith('.csv'):
                df = pd.read_csv(file_name, sep=';')
                letters = df['Rhyme Scheme Letter']
                data.append(letters)
        return data

    def print_data_statistics(self, data):
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

    def convert_data_to_int(self, data):
        for example in data:
            counts = Counter(example).items()
            # TODO

    def prepare_data_for_learning(self, true_data, test_split=0.3):
        # Generate false examples.
        false_examples = []
        for example in true_data:
            false_ex = np.random.sample(example)
            false_examples.append(false_ex)
        # Add labels.
        self.data = true_data + false_examples
        self.labels = [1]*len(true_data) + [0]*len(false_examples)
        # Shuffle.
        np.random.seed(42)
        self.data = np.random.sample(self.data)
        np.random.seed(42)
        self.labels = np.random.sample(self.labels)
        # Separate into batches.
        self.seq_groups = [[self.data[x % len(data)] for x in range(i, i + self.batch_size)] for i in range(0, len(self.data), self.batch_size)]
        self.label_groups = [[self.labels[x % len(self.labels)] for x in range(i, i + self.batch_size)] for i in range(0, len(self.labels), self.batch_size)]


