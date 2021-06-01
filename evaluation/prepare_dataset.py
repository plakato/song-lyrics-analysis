import argparse
import json
import random

from song_data.preprocess_data import save_dataset


def split_dataset(args):
    with open(args.file) as file:
        data = json.load(file)
    random.shuffle(data)
    l = len(data)
    train_idx = int(args.split*l)
    test_idx = train_idx + int(args.split*l)
    train_data = data[:train_idx]
    test_data = data[train_idx:min(test_idx, l)]
    save_dataset(train_data, f'data/train_lyrics{args.split}.json')
    save_dataset(test_data, f'data/test_lyrics{args.split}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help="A file for train file generation.")
    parser.add_argument('--split', type=float, help="Data of split proportion are extracted from file for training/testing.")

    args = parser.parse_args(['--file', '../song_data/data/ENlyrics_final.json',
                              '--split', '0.01'])
    split_dataset(args)
