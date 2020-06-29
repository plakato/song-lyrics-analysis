#!/usr/bin/env python3
import sys

import numpy as np
import tensorflow as tf

# from tensorflow.python import keras


# Custom filter
from generator import RhymeMatrixImageBatchGenerator
n_filters = 1


def cnn_filter(shape, dtype=None):
    n = n_filters
    f = np.array([[[[1]*n], [[1]*n], [[1]*n]],
                  [[[0]*n], [[0]*n], [[0]*n]],
                  [[[-1]*n], [[-1]*n], [[-1]*n]]
                  ])
    print(shape, f.shape)
    assert f.shape == shape
    return tf.keras.backend.variable(f, dtype='float32')


# The neural network model
class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()
        # regularizer = tf.keras.regularizers.l2(1e-4)
        inputs = tf.keras.layers.Input(shape=[None, None, 1])
        conv = tf.keras.layers.Conv2D(n_filters,  3, strides=2,
                                            padding='valid',
                                            kernel_initializer=cnn_filter,
                                            trainable=True,  # Don't change the filter.
                                            # kernel_regularizer=regularizer,
                                            data_format="channels_last",
                                            activation=tf.nn.relu)
        x = conv(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Optionally add more layers if needed.
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        # Optionally repeat the whole block more times.

        predict = tf.keras.layers.Activation('sigmoid')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=predict)
        self.model.compile(# optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        print(self.model.summary())

    def train(self, train_generator, val_generator, args):
        print_weights = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch,
                                                       logs: self.print_logs())
        history = self.model.fit(train_generator,
                                 epochs=args.epochs,
                                 verbose=1,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(args.logdir, monitor='val_loss', save_best_only=True, verbose=1),
                                           print_weights],
                                 validation_data=val_generator)
        return history

    def print_logs(self):
        print(self.model.layers[1].get_weights())
        print(self.model.output)

        # # With and without the activation
        # fun_without = tf.keras.backend.function([self.model.layers[0].input], [self.model.layers[0].output])
        # fun_with = tf.keras.backend.function([self.model.layers[0].input], [self.model.layers[1].output])
        # # Input
        # x_inp = np.reshape([], (1, 28, 28, 1))
        # # Output
        # layer_output_without = fun_without([x_inp])[0]
        # layer_output_with = fun_with([x_inp])[0]


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
    train_generator = RhymeMatrixImageBatchGenerator(train_dir, args.batch_size, shuffle_images=True, image_min_side=1)
    val_generator = RhymeMatrixImageBatchGenerator(val_dir, args.batch_size, shuffle_images=True, image_min_side=1)

    # Create the network and train.
    network = Network(args)
    network.train(train_generator, val_generator, args)

    # # Generate test set annotations, but in args.logdir to allow parallel execution.
    # with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
    #     for probs in network.predict(data['test']["images"],
    #                                  batch_size=args.batch_size):
    #         print(np.argmax(probs), file=out_file)
