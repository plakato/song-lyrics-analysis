from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.data import Dataset


class OneHotEncodingLayer(layers.experimental.preprocessing.PreprocessingLayer):
    def __init__(self, vocab_path=None, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        # max_tokens is vocab_size + 2
        self.vectorization = layers.experimental.preprocessing.TextVectorization(max_tokens=43,
                                                                                 output_mode="int",
                                                                                 output_sequence_length=1,
                                                                                 pad_to_max_tokens=True)
        if vocab_path:
            # Load precomputed vocabulary for TextVectorization layer.
            vocab = []
            with open(vocab_path) as vocab_file:
                lines = vocab_file.readlines()[:-1]
            for line in lines:
                vocab.append(line.strip())
            self.vectorization.set_vocabulary(vocab=vocab)
            # print(vectorize_layer.call([['a']]))
            # print(vectorize_layer.get_vocabulary())
            self.depth = len(vocab)
            indices = [i[0] for i in self.vectorization.call([[v] for v in vocab]).numpy()]
            self.minimum = min(indices)

    def adapt(self, data):
        self.vectorization.adapt(data)
        vocab = self.vectorization.get_vocabulary()
        self.depth = len(vocab)
        indices = [i[0] for i in self.vectorization([[v] for v in vocab]).numpy()]
        self.minimum = min(indices)

    def call(self, inputs):
        vectorized = self.vectorization.call(inputs)
        subtracted = tf.subtract(vectorized, tf.constant([self.minimum], dtype=tf.int64))
        encoded = tf.one_hot(subtracted, self.depth)
        return layers.Reshape((self.depth,))(encoded)

    def get_config(self):
        return {'vocabulary': self.vectorization.get_vocabulary(), 'depth': self.depth, 'minimum': self.minimum}
