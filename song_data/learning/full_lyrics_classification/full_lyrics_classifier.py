import numpy as np
import tensorflow as tf
from matplotlib import pyplot

from tensorflow.python import keras


# The neural network model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataset import get_dataset
from tensorflow.keras import layers


class Network(tf.keras.Sequential):
    def __init__(self, args):
        super().__init__()

        encoded_input = tf.keras.Input(shape=(None,), dtype=tf.int32, name='lyrics')
        # Set up preprocessing layer.
        x = layers.Embedding(args.vocab_size, 32)(encoded_input)
        x = layers.LSTM(args.lstm, return_sequences=True)(x)
        # x = layers.Dropout(args.dropout)(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(encoded_input, predictions)
        adam = tf.keras.optimizers.Adam(lr=args.lr)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
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
        print('Accuracy history:', history.history['accuracy'])
        return history

    def print_logs(self):
        print('Layer 2 weights:', self.model.layers[1].get_weights())
        print('Layer 3 weights:', self.model.layers[2].get_weights())
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
        pyplot.show()


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32
                        , type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout rate, LSTM layer.")
    parser.add_argument("--lstm", default=15, type=int, help="Neurons in LSTM layer.")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--vocab_size", default=32991, type=int, help="Size of vocabulary of SubwordTextEncoder used on input.")
    parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
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
    history = network.train(train_ds, val_ds, args)
    network.plot(history)
