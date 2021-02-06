import random

import tensorflow as tf
from tensorflow.python.data import Dataset
import tensorflow_datasets as tfds
import numpy as np


def get_dataset_encoded(dir='train', batch_size=32):
    # Load encoder.
    encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file('vocab')
    print('Vocab size is', encoder.vocab_size)
    # Load data.
    with open('dataset/' + dir + '/original.txt') as original:
        # Remove newline at the end.
        data_orig = original.readlines()[:-1]
    with open('dataset/' + dir + '/shuffled.txt') as shuffled:
        data_shuffled = shuffled.readlines()[:-1]
    data = data_orig + data_shuffled
    # Get song with max length to know the size for padding.
    max_len = 0
    longest_song = ''
    count = 0
    for i in range(len(data)):
        count += 1
        data[i] = data[i].strip()
        song = data[i]
        data[i] = encoder.encode(data[i])
        if len(data[i]) > max_len:
            max_len = len(data[i])
            longest_song = song
    print('max len is', max_len)
    print('longest song:', longest_song)
    # Create labels.
    labels = [1]*len(data_orig) + [0]*len(data_shuffled)
    # Shuffle.
    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(labels)
    # Create Dataset objects from generators.
    data_gen = lambda: (d for d in data)
    label_gen = lambda: ([l] for l in labels)
    dataset_data = tf.data.Dataset.from_generator(data_gen, output_types=tf.int32, output_shapes=tf.TensorShape([None]))
    dataset_labels = tf.data.Dataset.from_generator(label_gen, output_types=tf.int32, output_shapes=tf.TensorShape([1]))
    dataset = Dataset.zip((dataset_data, dataset_labels))
    # Each batch is padded to the size of the longest element in that batch.
    dataset_batched = dataset.padded_batch(batch_size, padding_values=0, padded_shapes=(max_len, 1))
    # Debug prints:
    print('{0} dataset: {1}'.format(dir, dataset_batched.cardinality()))
    # for element in dataset:
    #   print(element)
    for text_batch, label_batch in dataset_batched.take(1):
        print(text_batch.shape)
        print(label_batch.shape)
        for i in range(5):
            print(text_batch[i])
            print(label_batch[i])
    return dataset


def get_dataset_for_BERT(dir):
    # Load data.
    with open('dataset/' + dir + '/original.txt') as original:
        # Remove newline at the end.
        data_orig = original.readlines()[:-1]
    with open('dataset/' + dir + '/shuffled.txt') as modified:
        data_modified = modified.readlines()[:-1]
    data = data_orig + data_modified
    # Remove newlines.
    for i in range(len(data)):
        data[i] = data[i].strip().replace('>', '')
    # Create labels.
    labels = [1] * len(data_orig) + [0] * len(data_modified)
    return data

    return dataset


if __name__ == '__main__':
    # dataset = get_dataset_encoded()
    dataset = get_dataset_for_BERT('train')
