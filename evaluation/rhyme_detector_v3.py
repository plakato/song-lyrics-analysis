import json
import random
import re

import sklearn

from evaluation.rhyme_detector_v1 import get_pronunciations_for_n_syllables
from song_data.preprocess_data import save_dataset


def split_dataset(dataset_file, train_perc, test_perc):
    with open(dataset_file) as file:
        data = json.load(file)
    random.shuffle(data)
    l = len(data)
    train_idx = int(train_perc*l)
    test_idx = train_idx + int(test_perc*l)
    train_data = data[:train_idx]
    test_data = data[train_idx:min(test_idx, l)]
    save_dataset(train_data, f'data/train_lyrics{train_perc}.json')
    save_dataset(test_data, f'data/test_lyrics{test_perc}.json')


def extract_relevant_data(dataset_file):
    unique_cons_components = set()
    unique_vow_components = set()
    # In new data, every song is an array of possible pronunciations of last 4 syllables.
    new_data = []
    with open(dataset_file) as file:
        data = json.load(file)
        for song in data:
            lyrics_last4 = []
            for line in song['lyrics']:
                if line == '':
                    lyrics_last4.append([])
                    continue
                prons = get_pronunciations_for_n_syllables(line, 4)
                # Add all components (without stress) to sets of unique components.
                for pron in prons:
                    for syll in pron:
                        c1,v,c2 = syll
                        unique_cons_components.add(re.sub(r'\d+', '', ' '.join(c1).strip()))
                        unique_cons_components.add(re.sub(r'\d+', '', ' '.join(c2).strip()))
                        unique_vow_components.add(re.sub(r'\d+', '', ' '.join(v).strip()))
                lyrics_last4.append(prons)
            new_data.append(lyrics_last4)
    return new_data, unique_cons_components, unique_vow_components


def initialize_matrix(dim):
    matrix = [len(dim), len(dim)]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                matrix[i,j] = 1.0
            else:
                matrix[i,j] = 0.2
    return matrix


def first_step(data, matrixC, matrixV):
    pass

def find_rhymes(data, matrixC, matrixV):
    pass
def adjust_matrix(matrixC, matrixV, stats):
    pass


def main():
    split_dataset('../song_data/data/ENlyrics_final.json', 0.001, 0.001)
    data, cons_comp, vow_comp = extract_relevant_data('data/train_lyrics0.001.json')
    matrixC = initialize_matrix(cons_comp)
    matrixV = initialize_matrix(vow_comp)
    matrixC, matrixV = first_step(data, matrixC, matrixV)
    n = 0
    while n < 10:
        matrix = adjust_matrix(matrixC, matrixV, stats)
        stats = find_rhymes(data, matrixC, matrixV)
        n += 1


if __name__=='__main__':
    main()
