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
    # for i in range(len(data)):
    #     data[i] += [0]*(max_len - len(data[i]))
    # Add labels.
    labels = [1]*len(data_orig) + [0]*len(data_shuffled)
    # Shuffle.
    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(labels)
    # Convert to tensors.
    # data_tensor = tf.ragged.constant(data)
    # labels_tensor = tf.ragged.constant(labels)
    # # Convert to Dataset object.
    # features_dataset = Dataset.from_tensor_slices(data_tensor)
    # labels_dataset = Dataset.from_tensor_slices(labels_tensor)
    # dataset = Dataset.zip((features_dataset, labels_dataset))
    # # Convert to numpy array to create Dataset object.
    # dataset = Dataset.from_tensor_slices((data, labels))
    data_gen = lambda: (d for d in data)
    label_gen = lambda: ([l] for l in labels)
    dataset_data = tf.data.Dataset.from_generator(data_gen, output_types=tf.int32, output_shapes=tf.TensorShape([None]))
    dataset_labels = tf.data.Dataset.from_generator(label_gen, output_types=tf.int32, output_shapes=tf.TensorShape([1]))
    dataset = Dataset.zip((dataset_data, dataset_labels))
    # im_dataset = im_dataset.prefetch(4)
    # print("output data type is ", im_dataset.output_types)
    # print("output data shape is ", im_dataset.output_shapes)
    # iterator = im_dataset.make_initializable_iterator()
    # with tf.Session() as sess:
    #     sess.run(iterator.initializer)
    #     a = sess.run(iterator.get_next())
    # print("shape of the run results are: ")
    # print(a[0].shape)
    # print(a[1].shape)
    # print(a[2].shape)
    # print(a[3].shape)
    # for elem, val in dataset:
    #     print(elem)
    #     print(val)
    #     break
    # Each batch is padded to the size of the longest element in that batch.
    dataset_batched = dataset.padded_batch(batch_size, padding_values=0, padded_shapes=(max_len, 1))
    # Debug prints:
    print('{0} dataset: {1}'.format(dir, dataset_batched.cardinality()))
    # for element in dataset:
    #   print(element)
    # for text_batch, label_batch in dataset_batched.take(1):
    #     print(text_batch.shape)
    #     print(label_batch.shape)
    #     for i in range(5):
    #         print(text_batch[i])
    #         print(label_batch[i])
    return dataset


def set_shapes(image, label):
    image.set_shape((300, 300, 3))
    label.set_shape([])
    return image, label

if __name__ == '__main__':
    dataset = get_dataset()
