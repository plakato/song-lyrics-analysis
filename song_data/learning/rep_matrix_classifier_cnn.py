#!/usr/bin/env python3
import random
from os.path import isfile, join

import numpy as np
import tensorflow as tf

from tensorflow.python import keras
import sklearn
from sklearn.model_selection import train_test_split


# Custom filter
from song_data.learning.generator import Generator


def cnn_filter(shape, dtype=None):
    f = np.array([[[[1]], [[1]], [[1]]],
                  [[[0]], [[0]], [[0]]],
                  [[[-1]], [[-1]], [[-1]]]
                  ])
    # f = np.array([
    #         [[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]],
    #         [[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]],
    #         [[[-1], [-1], [-1]], [[-1], [-1], [-1]], [[-1], [-1], [-1]]]
    #     ])
    print(shape, f.shape)
    assert f.shape == shape
    return keras.backend.variable(f, dtype='float32')


# The neural network model
class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()
        regularizer = tf.keras.regularizers.l2(1e-4)
        inputs = tf.keras.layers.Input(shape=[None, None, 1])
        conv_layer = tf.keras.layers.Conv2D(1,  3,
                                            strides=(3,3), padding='valid',
                                            kernel_initializer=cnn_filter,
                                            trainable=False,  # Don't change the
                                            # filter.
                                            kernel_regularizer=regularizer,
                                            data_format="channels_last",
                                            activation=tf.nn.relu)(inputs)
        normalized = tf.keras.layers.BatchNormalization()(conv_layer)
        # Optionally add more layers if needed.
        pooled = tf.keras.layers.GlobalMaxPooling2D()(normalized)
        # Optionally repeat the whole block more times.

        predict = tf.keras.layers.Activation('softmax')(pooled)

        self.model = tf.keras.Model(inputs=inputs, outputs=predict)
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        print(self.model.summary())

    def train(self, train_generator, val_generator, args):
        history = self.model.fit(train_generator,
                                 epochs=args.epochs,
                                 verbose=1,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(args.logdir, monitor='val_loss', save_best_only=True, verbose=1)],
                                 validation_data=val_generator)
        return history


def load_images_for_dir(dir):
    path = args.data_dir + dir
    images = []
    for file in os.listdir(path):
        image = tf.keras.preprocessing.image.load_img(path + file, color_mode="grayscale")
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        images.append(input_arr)
    labels = [np.float(1)] * len(images)
    return images, labels


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Batch "
                                                                  "size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--data_dir",
                        default="dataset",
                        type=str,
                        help="Path to input data with images.")
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

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    train_generator = Generator(train_dir, args.batch_size, shuffle_images=True, image_min_side=1)
    val_generator = Generator(val_dir, args.batch_size, shuffle_images=True, image_min_side=1)

    # Create the network and train.
    network = Network(args)
    network.train(train_generator, val_generator, args)

    # # Generate test set annotations, but in args.logdir to allow parallel execution.
    # with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
    #     for probs in network.predict(data['test']["images"],
    #                                  batch_size=args.batch_size):
    #         print(np.argmax(probs), file=out_file)
