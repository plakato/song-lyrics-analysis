import random

import tensorflow as tf
from tensorflow.python.data import Dataset
import tensorflow_datasets as tfds
import numpy as np


def get_dataset(dir='train', batch_size=32):
    # Load encoder.
    encoder = tfds.features.text.SubwordTextEncoder.load_from_file('vocab')
    # Load data.
    with open('dataset/' + dir + '/original.txt') as original:
        # Remove newline at the end.
        data_orig = original.readlines()[:-1]
    with open('dataset/' + dir + '/shuffled.txt') as shuffled:
        data_shuffled = shuffled.readlines()[:-1]
    data = data_orig + data_shuffled
    max_len = 0
    count = 0
    for i in range(len(data)):
        count += 1
        data[i] = data[i].strip()
        data[i] = encoder.encode(data[i])
        if len(data[i]) > max_len:
            max_len = len(data[i])
    print('max len is', max_len)
    # Add padding.
    for i in range(len(data)):
        data[i] += [0]*(max_len - len(data[i]))
    # Add labels.
    labels = [1]*len(data_orig) + [0]*len(data_shuffled)
    # Shuffle.
    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(labels)
    # Make batches.
    batched_data = np.asarray([np.asarray(data[i:i+batch_size]) for i in range(0, len(data), batch_size)])
    batched_labels = np.asarray([np.asarray(labels[i:i+batch_size]) for i in range(0, len(labels), batch_size)])
    return batched_data, batched_labels


if __name__ == '__main__':
    dataset = get_dataset()
