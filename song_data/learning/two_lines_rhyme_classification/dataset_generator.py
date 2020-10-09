import math
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from prepare_dataset import epoch_sep
import tensorflow_datasets as tfds
import tensorflow as tf


class DataGenerator(Sequence):
    def __init__(self, dataset_path, batch_size=16):
        'Initialization'
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        # Pointer to a file from which we read the dataset on demand.
        self.rhyming_stream_pointer = 0
        self.not_rhyming_stream_pointer = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]
        # Generate data
        first_verses = [first for first, _, _ in self.dataset[idxs]]
        second_verses = [second for _, second, _ in self.dataset[idxs]]
        y = [y for _, _, y in self.dataset[idxs]]
        return [first_verses, second_verses],[y]

    def load_epoch(self, filename, pointer, label):
        encoder = tfds.features.text.SubwordTextEncoder.load_from_file('vocab')
        print('vocab size:', encoder.vocab_size)
        pairs = []
        with open(filename, 'r') as rhyming_file:
            rhyming_file.seek(pointer)
            first_verse = rhyming_file.readline()
            while first_verse != epoch_sep:
                second_verse = rhyming_file.readline()
                first_verse = encoder.encode(first_verse.strip())
                second_verse = encoder.encode(second_verse.strip())
                pairs.append([tf.convert_to_tensor(first_verse, dtype=tf.int32),
                              tf.convert_to_tensor(second_verse, dtype=tf.int32),
                              tf.convert_to_tensor([label], dtype=tf.int32)])
                # pairs.append([first_verse, second_verse, label])
                first_verse = rhyming_file.readline()
            new_pointer = rhyming_file.tell()
        return pairs, new_pointer

    def on_epoch_end(self):
        'Loads new data after each epoch'
        # Label rhyming with 0 and not rhyming as 1.
        rhyming, self.rhyming_stream_pointer = self.load_epoch(self.dataset_path+'/rhyming.txt', self.rhyming_stream_pointer, 1)
        not_rhyming, self.not_rhyming_stream_pointer = self.load_epoch(self.dataset_path+'/not_rhyming.txt', self.not_rhyming_stream_pointer, 0)
        self.dataset = np.array(rhyming + not_rhyming)
        np.random.shuffle(self.dataset)


if __name__ == '__main__':
    gen = DataGenerator('dataset/train')
    batch = gen.__getitem__(0)
    print("f,s,y:", batch)

