import numpy as np
import tensorflow as tf

from tensorflow.python import keras


# The neural network model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataset import get_dataset
from tensorflow.keras import layers
from tensorflow.python.data import Dataset


class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()

        text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')

        max_features = 43
        sequence_length = 275
        # Set up preprocessing layer.
        vectorize_layer = TextVectorization(max_tokens=max_features,
                                            output_mode="int",
                                            output_sequence_length=sequence_length,
                                            pad_to_max_tokens=True)
        # Load precomputed vocabulary for TextVectorization layer.
        vocab = []
        with open(args.vocab_path) as vocab_file:
            lines = vocab_file.readlines()[:-1]
        for line in lines:
            vocab.append(line.strip())
        vectorize_layer.set_vocabulary(vocab=vocab)
        x = vectorize_layer(text_input)
        x = layers.Embedding(max_features + 1, output_dim=10, input_length=sequence_length)(x)
        # x = layers.Dropout(0.1)(x)
        x = layers.LSTM(10, dropout=0.2, recurrent_dropout=0.2)(x)
        # x = layers.Dropout(0.2)(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(text_input, predictions)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

        # raw_train_ds = get_dataset(dir='train', batch_size=32)
        # # Make a text-only dataset (no labels):
        # text_ds = raw_train_ds.map(lambda x, y: x)
        # for text_batch in text_ds.take(1):
        #     for i in range(5):
        #         print(text_batch.numpy()[i])
        # # text_ds = list(text_ds.as_numpy_iterator())
        # # Adapt to create vocabulary.
        # print(text_ds)
        #
        #     vectorize_layer.adapt(text_ds)

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        print(self.model.summary())

    def train(self, train_ds, val_ds, args):
        history = self.model.fit(train_ds,
                                 epochs=args.epochs,
                                 verbose=1,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(args.logdir, monitor='val_loss', save_best_only=True, verbose=1), self.tb_callback],
                                 validation_data=val_ds)
        return history

    # Load precomputed vocabulary for TextVectorization layer.
    def load_vocab(self):
        vocab = []
        with open(args.vocab_path) as vocab_file:
            lines = vocab_file.readlines()[:-1]
        for line in lines:
            vocab.append(line.strip())
        return vocab


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Batch "
                                                                  "size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--vocab_path",
                        default="dataset/vocab.txt",
                        type=str,
                        help="Path to precomputed vocabulary.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    train_ds = get_dataset(dir='train', batch_size=args.batch_size)
    val_ds = get_dataset(dir='val', batch_size=args.batch_size)
    # Create the network and train.
    network = Network(args)
    network.train(train_ds, val_ds, args)
