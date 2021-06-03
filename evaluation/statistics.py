# Percentage of line-end words that are not in CMU dictionary.
import argparse
import json
import os
import re

import cmudict


def percetage_not_in_CMU(filename, verbose):
    total_words = 0
    n_not_in_CMU = 0
    dict = cmudict.dict()
    total_songs = 0
    if verbose:
        print('Following words are not in CMU dictionary:')
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        for song in data:
            if verbose:
                print(f"SONG: {song['title']}")
            total_songs += 1
            for line in song['lyrics']:
                last_word = line.strip().split(' ')[-1]
                last_word = last_word.lower()
                # Remove punctuation.
                last_word = re.sub(r"[^\w\d'\s]+", '', last_word)
                if last_word == '':
                    continue
                pron = dict.get(last_word)
                if not pron and verbose:
                    print(last_word)
                    n_not_in_CMU += 1
                total_words += 1
    print(f'Analyzed {total_songs} songs.')
    return n_not_in_CMU/total_words


# Returns
# - how many percent of line-end words contain a number
# - how many percent of line-end words consist of only numbers
def percentage_of_numbers_to_all_OOD(filename, verbose):
    n_not_in_CMU = 0
    n_only_numbers = 0
    n_contain_numbers = 0
    only_numbers = []
    contain_numbers = []
    dict = cmudict.dict()
    total_songs = 0
    if verbose:
        print('Following words are not in CMU dictionary and ARE NOT NUMBERS:')
    with open(filename, 'r') as input_file:
        data = json.load(input_file)
        for song in data:
            total_songs += 1
            for line in song['lyrics']:
                last_word = line.strip().split(' ')[-1]
                last_word = last_word.lower()
                # Remove punctuation.
                last_word = re.sub(r"[^\w\d'\s]+", '', last_word)
                if last_word == '':
                    continue
                pron = dict.get(last_word)
                if not pron:
                    if last_word.isdecimal():
                        n_only_numbers += 1
                        only_numbers.append(last_word)
                    elif re.search("\d", last_word):
                        n_contain_numbers += 1
                        contain_numbers.append(last_word)
                    else:
                        if verbose:
                            print(last_word)
                    n_not_in_CMU += 1
    if verbose:
        print("Following words contain a number:")
        for n in contain_numbers:
            print(n)
        print("Following words are numbers:")
        for n in only_numbers:
            print(n)
    print(f'Analyzed {total_songs} songs.')
    return (n_contain_numbers+n_only_numbers)/n_not_in_CMU, n_only_numbers/n_not_in_CMU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_in_cmu', default=False, action='store_true', help="Prints the percentage of line-end word that are not in CMU.")
    parser.add_argument('--numbers_in_OOD', default=False, action='store_true', help="Prints statistics about numbers among out-of-dictionary words.")
    parser.add_argument('--file', required=True, help="JSON file with songs to calculate the statistics for.")
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args(['--numbers_in_OOD', '--file', '../song_data/data/ENlyrics_final.json'])
    if args.not_in_cmu:
        percentage = percetage_not_in_CMU(args.file, args.verbose)
        print(f'{percentage*100}% of line-end words are not in CMU dictionary.')
    if args.numbers_in_OOD:
        contain_number, only_numbers = percentage_of_numbers_to_all_OOD(args.file, args.verbose)
        print(f"{contain_number*100}% of out-of-dictionary words contain numbers.")
        print(f"{only_numbers*100}% of out-of-dictionary words are numbers.")