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
    for i in range(len(data)):
        data[i] = data[i].strip()
        data[i] = encoder.encode(data[i])
    # Add labels.
    labels = [1]*len(data_orig) + [0]*len(data_shuffled)
    # Shuffle.
    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(labels)
    # Convert to tensors.
    data_tensor = tf.ragged.constant(data)
    labels_tensor = tf.ragged.constant(labels)
    # Convert to Dataset object.
    features_dataset = Dataset.from_tensor_slices(data_tensor)
    labels_dataset = Dataset.from_tensor_slices(labels_tensor)
    dataset = Dataset.zip((features_dataset, labels_dataset))
    # Convert to numpy array to create Dataset object.
    # dataset = Dataset.from_tensor_slices((data, labels))
    # # Each batch is padded to the size of the longest element in that batch.
    dataset = dataset.padded_batch(batch_size, padding_values=0)
    # Debug prints:
    print('{0} dataset: {1}'.format(dir, dataset.cardinality()))
    # for element in dataset:
    #   print(element)
    for text_batch, label_batch in dataset.take(1):
        for i in range(5):
            print(text_batch.numpy()[i])
            print(label_batch.numpy()[i])
    return dataset


if __name__ == '__main__':
    dataset = get_dataset()
