#!/usr/bin/env python3
import random
from os.path import isfile, join

import numpy as np
import tensorflow as tf

from tensorflow.python import keras
import sklearn
from sklearn.model_selection import train_test_split


# Custom filter
def cnn_filter(shape, dtype=None):

    f = np.array([
            [[[1]], [[1]], [[1]]],
            [[[0]], [[0]], [[0]]],
            [[[-1]], [[-1]], [[-1]]]
        ])
    print(shape)
    assert f.shape == shape
    return keras.backend.variable(f, dtype='float32')


# The neural network model
class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()
        regularizer = tf.keras.regularizers.l2(1e-4)
        self.add(tf.keras.layers.Conv2D(1, (3, 3), padding='valid',
                                        kernel_initializer=cnn_filter,
                                        trainable=False,  # Don't change the
                                        # filter.
                                        kernel_regularizer=regularizer,
                                        data_format="channels_last",
                                        input_shape=(None, None, 1)))
        self.add(tf.keras.layers.Activation('elu'))
        self.add(tf.keras.layers.BatchNormalization())
        # Optionally add more layers if needed.
        self.add(tf.keras.layers.GlobalMaxPooling2D())
        # Optionally repeat the whole block more times.

        # self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, data, args):
        self.fit(
            data['train']["images"], data['train']["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(data['test']["images"], data['test']["labels"]),
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
    parser.add_argument("--batch_size", default=1, type=int, help="Batch "
                                                                  "size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--data_dir",
                        default="../sparsar_experiments/repetition_matrices/",
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

    # Load data
    # data = tf.keras.preprocessing.image_dataset_from_directory(args.data_dir,
    #     labels="inferred", label_mode="int", class_names=None,
    #     color_mode="rgb", batch_size=32, image_size=[None, None],
    #                                                            shuffle=False,
    #     seed=123, validation_split=0.2, subset='training',
    #     interpolation="bilinear", follow_links=False, )
    # Load images.
    path = args.data_dir + 'endrhymes_original/'
    images = []
    for file in os.listdir(path):
        image = tf.keras.preprocessing.image.load_img(path+file)
        input_arr = keras.preprocessing.image.img_to_array(image)
        images.append(input_arr)
    labels = [np.float(1)] * len(images)
    all_images = images
    all_labels = labels
    path = args.data_dir + 'endrhymes_shuffled/'
    images = []
    for file in os.listdir(path):
        image = tf.keras.preprocessing.image.load_img(path+file)
        input_arr = keras.preprocessing.image.img_to_array(image)
        images.append(input_arr)
    labels = [np.float(0)] * len(images)
    all_images.append(images)
    all_labels.append(labels)
    # Split to train/test.
    idxs = range(len(all_labels))
    test_count = int(len(all_labels) * 0.2)
    test_idxs = random.sample(idxs, test_count)
    images_train, images_test, labels_train, labels_test = train_test_split(all_images, all_labels, test_size=0.2)
    tensor = tf.convert_to_tensor(np.array(labels_train, dtype=object))
    # # Create batches.
    # images_train = [images_train[i:i + args.batch_size] for i in range(0,
    #                                                        len(images_train),
    #                                                      args.batch_size)]
    # labels_train = [labels_train[i:i + args.batch_size] for i in
    #                 range(0, len(labels_train), args.batch_size)]
    data = {'train': {'images': np.array(images_train),
                      'labels': np.array(labels_train)},
            'test': {'images': np.array(images_test),
                     'labels': np.array(labels_test)}}
    # Create the network and train.
    network = Network(args)
    network.train(data, args)

    # # Generate test set annotations, but in args.logdir to allow parallel execution.
    # with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
    #     for probs in network.predict(data['test']["images"],
    #                                  batch_size=args.batch_size):
    #         print(np.argmax(probs), file=out_file)
