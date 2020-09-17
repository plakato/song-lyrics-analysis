import numpy as np
import tensorflow as tf

from tensorflow.python import keras


# The neural network model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataset import get_dataset
from tensorflow.keras import layers


class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()

        encoded_input = tf.keras.Input(shape=(1, ), dtype=tf.string, name='text')
        # Set up preprocessing layer.
        x = layers.Embedding(1000, 5)(encoded_input)
        x = layers.LSTM(args.lstm)(x)
        x = layers.Dropout(args.dropout)(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(encoded_input, predictions)
        adam = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

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


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64
                        , type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.1, type=int, help="Dropout rate, LSTM layer.")
    parser.add_argument("--lstm", default=100, type=int, help="Neurons in LSTM layer.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
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
