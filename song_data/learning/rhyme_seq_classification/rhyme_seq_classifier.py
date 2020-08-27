import numpy as np
import tensorflow as tf

from tensorflow.python import keras


# The neural network model
class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()
        max_words = 5000
        embedding_vecor_length = 32
        self.model.add(keras.layers.embeddings.Embedding(max_words, embedding_vecor_length, input_length=max_review_length))
        self.model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

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
        self.model.fit(X_train, y_train, epochs=3, batch_size=64)
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))


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
                        default="dataset_big",
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
