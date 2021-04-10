# Percentage of line-end words that are not in CMU dictionary.
import os
import re

import cmudict


def percetage_not_in_CMU(directory):
    total = 0
    n_not_in_CMU = 0
    dict = cmudict.dict()
    print('Following words are not in CMU dictionary:')
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            print(directory+filename)
            with open(directory+filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    last_word = line.strip().split(' ')[-1]
                    last_word = last_word.lower()
                    # Remove punctuation.
                    last_word = re.sub(r"[^\w\d'\s]+", '', last_word)
                    if last_word == '':
                        continue
                    pron = dict.get(last_word)
                    if not pron:
                        print(last_word)
                        n_not_in_CMU += 1
                    total += 1
    return n_not_in_CMU/total


if __name__ == '__main__':
    percentage = percetage_not_in_CMU('../song_data/data/individual_songs/')
    print(f'{percentage*100}% of line-end words are not in CMU dictionary.')