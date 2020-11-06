import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import tensorflow_datasets as tfds

from song_data.learning.two_lines_rhyme_classification.dataset_generator import DataGenerator


# The neural network model
class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()

        first_verse = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='first')
        second_verse = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='second')
        # Set up preprocessing layer.
        first_embedded = tf.keras.layers.Embedding(args.vocab_size, 32)(first_verse)
        second_embedded = tf.keras.layers.Embedding(args.vocab_size, 32)(second_verse)
        concats = tf.keras.layers.Concatenate(axis=1)([first_embedded, second_embedded])
        x = tf.keras.layers.LSTM(args.lstm, return_sequences=True)(concats)
        x = tf.keras.layers.LSTM(args.lstm, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(args.dropout)(x)
        x = tf.keras.layers.LSTM(args.lstm, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(args.lstm)(x)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.models.Model([first_verse, second_verse], predictions)
        adam = tf.keras.optimizers.Adam(lr=args.lr)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        print(self.model.summary())

    def train(self, train_ds, val_ds, args):
        print_weights = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch,
                                                       logs: self.print_logs())
        history = self.model.fit(train_ds,
                                 epochs=args.epochs,
                                 verbose=1,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(args.logdir,
                                                                               monitor='val_loss',
                                                                               save_best_only=True,
                                                                               verbose=1),
                                            self.tb_callback,
                                            print_weights],
                                 validation_data=val_ds)
        print('Loss history:', history.history['loss'])
        print('Accuracy history:'
              '', history.history['accuracy'])
        return history

    def print_logs(self):
        print("Layer 1 weights:", self.model.layers[1].get_weights())
        print("Layer 2 weights:", self.model.layers[2].get_weights())
        print('Input:', self.model.input)
        print('Output:', self.model.output)

    def plot(self, history):
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig(args.logdir + '/loss.png')


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=512
                        , type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate, LSTM layer.")
    parser.add_argument("--lstm", default=100, type=int, help="Neurons in LSTM layer.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
    # parser.add_argument("--vocab_size", default=32659, type=int, help="Size of vocabulary of SubwordTextEncoder used on input + 1.")
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
    # Set vocabulary size.
    encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file('vocab')
    args.vocab_size = encoder.vocab_size
    train_ds = DataGenerator('dataset/train', batch_size=args.batch_size)
    val_ds = DataGenerator('dataset/val', batch_size=args.batch_size)
    # Create the network and train.
    network = Network(args)
    history = network.train(train_ds, val_ds, args)
    network.plot(history)
