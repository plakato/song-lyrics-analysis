#!/usr/bin/env python3
from os.path import isfile, join

import numpy as np
import tensorflow as tf

from tensorflow.python import keras


# Custom filter
def cnn_filter(shape, dtype=None):

    f = np.array([
            [[[1]], [[1]], [[1]]],
            [[[0]], [[0]], [[0]]],
            [[[-1]], [[-1]], [[-1]]]
        ])
    assert f.shape == shape
    return keras.backend.variable(f, dtype='float32')


# The neural network model
class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()
        regularizer = tf.keras.regularizers.l2(1e-4)
        self.add(tf.keras.layers.Conv2D(32, (3, 3), padding='valid',
                                        kernel_initializer=cnn_filter,
                                        trainable=False,  # Don't change the
                                        # filter.
                                        kernel_regularizer=regularizer,
                                        input_shape=(None, None, 1)))
        self.add(tf.keras.layers.Activation('elu'))
        self.add(tf.keras.layers.BatchNormalization())
        # Optionally add more layers if needed.
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(tf.keras.layers.Dropout(0.2))
        # Optionally repeat the whole block more times.

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(10, activation="softmax"))

        self.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, data, args):
        self.fit(
            data.train.data["images"], data.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(data.dev.data["images"], data.dev.data[
                "labels"]),
            callbacks=[self.tb_callback],
        )
        # self.save_weights('40epochs.h5')


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--data_dir",
                        default="../sparsar_experiments/rhymes/original/",
                        type=str,
                        help="Path to input data.")
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

    # Load data
    for item in os.listdir(args.data_path):
        if isfile(join(args.data_path, item)) and item.endswith('.png'):
            # TODO


    # Create the network and train
    network = Network(args)
    network.train(data, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
