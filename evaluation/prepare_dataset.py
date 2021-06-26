import argparse
import json
import random

from song_data.preprocess_data import save_dataset


def split_dataset(args):
    with open(args.file) as file:
        data = json.load(file)
    random.shuffle(data)
    l = len(data)
    train_idx = 0
    if args.train:
        train_idx = int(args.train*l)
        train_data = data[:train_idx]
        save_dataset(train_data, f'data/train_lyrics{args.train}.json')
    dev_idx = train_idx
    if args.dev:
        dev_idx = train_idx + int(args.dev * l)
        dev_data = data[train_idx:min(dev_idx, l)]
        save_dataset(dev_data, f'data/dev_lyrics{args.dev}.json')
    if args.test:
        test_idx = dev_idx + int(args.test*l)
        test_data = data[dev_idx:min(test_idx, l)]
        save_dataset(test_data, f'data/test_lyrics{args.test}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help="A file for train file generation.")
    parser.add_argument('--train', type=float, default=None, help="Proportion of data for training.")
    parser.add_argument('--dev', type=float, default=None, help="Proportion of data for dev.")
    parser.add_argument('--test', type=float, default=None, help="Proportion of data for testing.")

    args = parser.parse_args(['--file', '../song_data/data/ENlyrics_final.json',
                              '--train', '0.001',
                              '--dev', '0.001'])
    split_dataset(args)
