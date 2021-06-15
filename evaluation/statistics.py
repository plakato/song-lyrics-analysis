# Percentage of line-end words that are not in CMU dictionary.
import argparse
import json
import os
import re

import cmudict

from evaluation.constants import NO_OF_PRECEDING_LINES
from evaluation.rhyme_detector_v3 import RhymeDetector


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


# Returns top n co-occurrences of components (not-identical).
def print_nonidentical_cooccurrences(file, n, out_file):
    comp_coocurence = dict()
    separator = '&'
    rd = RhymeDetector()
    data = rd.load_and_preprocess_data_from_file(file)
    for song in data:
        for line_idx in range(1, len(song)):
            if not song[line_idx]:
                continue

            # Finds rhyme in preceding lines. Returns relevant components for both lines + flag if the rhyme was perfect.
            def find_rhyme_for_line():
                rel_first = []
                rel_second = []
                found_perfect = False
                for pronunciation1 in song[line_idx]:
                    # Look for rhyme fellow in preceding lines.
                    for lines_back in range(min(NO_OF_PRECEDING_LINES, line_idx), 0, -1):
                        if not song[line_idx - lines_back]:
                            continue
                        for pronunciation2 in song[line_idx - lines_back]:
                            rel_first, rel_second, _ = RhymeDetector.get_relevant_components_for_pair(pronunciation1, pronunciation2)
                            if rel_first == rel_second:
                                found_perfect = True
                                return rel_first, rel_second, found_perfect
                return rel_first, rel_second, found_perfect
            # Ignore if we found perfect rhyme or only "rhyme" with an empty line.
            rel_first, rel_second, found_perfect = find_rhyme_for_line()
            if found_perfect or rel_first == [] or rel_second == []:
                continue
            # Add components to cooccurrence list if they are not identical.
            for i in range(len(rel_first)):
                if rel_first[i] != rel_second[i]:
                    index = separator.join(sorted([rel_first[i], rel_second[i]]))
                    if index in comp_coocurence:
                        comp_coocurence[index] += 1
                    else:
                        comp_coocurence[index] = 1
    # Save in format that can be used by rhyme detector.
    if out_file:
        # Calculate individual frequencies.
        indiv_freq = dict()
        for key in comp_coocurence:
            indivs = key.split(rd.separator)
            for i in [0, 1]:
                if indivs[i] in indiv_freq:
                    indiv_freq[indivs[i]] += 1
                else:
                    indiv_freq[indivs[i]] = 1
        rd.cooc = rd.calculate_new_matrix_from_frequencies(indiv_freq, comp_coocurence)
        rd.save_matrix(out_file)
    else:
        # Sort the list of co-occurrences and print the top n combinations.
        for comp, occ in sorted(comp_coocurence.items(), key=lambda item: item[1], reverse=True):
            comp = comp.split(separator)
            print(f"{comp[0]:<5} | {comp[1]:<5}: {occ}")
            n -= 1
            if n == 0:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_in_cmu', default=False, action='store_true', help="Prints the percentage of line-end word that are not in CMU.")
    parser.add_argument('--numbers_in_OOD', default=False, action='store_true', help="Prints statistics about numbers among out-of-dictionary words.")
    parser.add_argument('--n_cooccurrences', default=False, type=int, help="Prints top n non-identical co-occurrences of components.")
    parser.add_argument('--cooc_file', default=None, help="File where co-occurrences in format that allows it to be used by rhyme_detector_v3.")
    parser.add_argument('--file', required=True, help="JSON file with songs to calculate the statistics for.")
    parser.add_argument('--verbose', default=False, action='store_true')
    # args = parser.parse_args(['--numbers_in_OOD', '--file', '../song_data/data/ENlyrics_final.json'])
    args = parser.parse_args(['--n_cooccurrences', '100',
                              '--file', 'data/train_lyrics0.01.json',
                              '--cooc_file', 'data/cooc_statistical_0.01.json'])
    if args.not_in_cmu:
        percentage = percetage_not_in_CMU(args.file, args.verbose)
        print(f'{percentage*100}% of line-end words are not in CMU dictionary.')
    if args.numbers_in_OOD:
        contain_number, only_numbers = percentage_of_numbers_to_all_OOD(args.file, args.verbose)
        print(f"{contain_number*100}% of out-of-dictionary words contain numbers.")
        print(f"{only_numbers*100}% of out-of-dictionary words are numbers.")
    if args.n_cooccurrences:
        print_nonidentical_cooccurrences(args.file, args.n_cooccurrences, args.cooc_file)
