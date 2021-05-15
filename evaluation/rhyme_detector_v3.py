import json
import math
import random
import re

import sklearn

from evaluation.constants import NO_OF_PRECEDING_LINES, NOT_AVAILABLE
from evaluation.rhyme_detector_v1 import get_pronunciations_for_n_syllables, next_letter_generator
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


# Returns an array of songs.
# Each song is an array of lines.
# Each line is an array possible pronunciations.
# Each pronunciation is an array of 4 last syllables.
# Each syllable is a triplet CVC.
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
    unique_cons_components = sorted(list(unique_cons_components))
    unique_vow_components = sorted(list(unique_vow_components))
    return new_data, unique_cons_components, unique_vow_components


def initialize_matrix(size):
    matrix = [[0.6]*size for i in range(size)]
    for i in range(len(matrix)):
        matrix[i][i] = 1.0
    return matrix


def first_step(data, matrixC, matrixV):
    pass


# Get components after the last stress.
def get_relevant_components(first, second):
    stress_penalty = False
    last_stressed_idx1 = 4
    last_stressed_idx2 = 4
    # Extract relevant components - everything from last stressed component.
    for i in range(len(first)):
        _, v, _ = first[i]
        v = ''.join(v)
        if '1' in v or '2' in 'v':
            last_stressed_idx1 = i
    for i in range(len(second)):
        _, v, _ = second[i]
        v = ''.join(v)
        if '1' in v or '2' in 'v':
            last_stressed_idx2 = i
    if len(first) - last_stressed_idx1 != len(second) - last_stressed_idx2:
        stress_penalty = True
    # Select index of last stressed syllable.
    relevent_last_sylls = max(len(first) - last_stressed_idx1, len(second) - last_stressed_idx2)
    if relevent_last_sylls > len(first):
        relevent_last_sylls = len(first)
    if relevent_last_sylls > len(second):
        relevent_last_sylls = len(second)
    rel_first = []
    rel_second = []
    for i in range(relevent_last_sylls, 0, -1):
        first_c1, first_v, first_c2 = first[-i]
        sec_c1, sec_v, sec_c2 = second[-i]
        if i == relevent_last_sylls:
            rel_first.extend([first_v, first_c2])
            rel_second.extend([sec_v, sec_c2])
        else:
            rel_first.extend([first_c1, first_v, first_c2])
            rel_second.extend([sec_c1, sec_v, sec_c2])
    rel_first = [re.sub(r'\d', '', ' '.join(item).strip()) for item in rel_first]
    rel_second = [re.sub(r'\d', '', ' '.join(item).strip()) for item in rel_second]
    return rel_first, rel_second, stress_penalty


def get_rhyme_rating(first, second, stress_penalty, mC, mV):
    rating = 1
    matrixC, cons_idx = mC
    matrixV, vow_idx = mV
    vowel = True
    next_vowel = False
    for i in range(len(first)):
        a, b = sorted([first[i], second[i]])
        if vowel:
            ai = vow_idx.index(a)
            bi = vow_idx.index(b)
            rating *= matrixV[ai][bi]
        else:
            ai = cons_idx.index(a)
            bi = cons_idx.index(b)
            rating *= matrixC[ai][bi]
        last_vowel = vowel
        vowel = next_vowel
        if vowel:
            next_vowel = False
        elif not last_vowel:
            next_vowel = True
    if stress_penalty:
        rating -= 0.1
    return rating


# Returns a list of song object.
# Song object contains: scheme, ratings, relevant_components.
def find_rhymes(data, matrixC, matrixV):
    stats = []
    for song in data:
        # Assign the letter for the first line -> rhyme can't be detected yet.
        letter_gen = next_letter_generator()
        scheme = ['']*len(song)
        ratings = [0]*len(song)
        start_idx = 0
        while not song[start_idx]:
            ratings[start_idx] = NOT_AVAILABLE
            start_idx += 1
        scheme[start_idx] = next(letter_gen)
        # Components participating in rhyme (or not).
        # Their matrix values will be recalculated later.
        relevant_comps = [None]*len(song)
        for line_idx in range(start_idx + 1, len(song)):
            # Find the best rated combination of pronunciation of this line and its selected predecessors.
            rhyme_fellows = []
            if not song[line_idx]:
                ratings[line_idx] = NOT_AVAILABLE
                continue
            for pronunciation1 in song[line_idx]:
                for lines_back in range(1, min(NO_OF_PRECEDING_LINES, line_idx) + 1):
                    for pronunciation2 in song[line_idx-lines_back]:
                        rel_first, rel_second, stress_penalty = get_relevant_components(pronunciation1, pronunciation2)
                        rating = get_rhyme_rating(rel_first, rel_second, stress_penalty, matrixC, matrixV)
                        if not relevant_comps[0] and line_idx-lines_back == 0:
                            relevant_comps[0] = rel_second
                        result = {'rating': rating,
                                  'relevant_comps1': rel_first,
                                  'relevant_comps2': rel_second,
                                  'index': line_idx - lines_back}
                        rhyme_fellows.append(result)
            # Select the rhyme fellow with best rating.
            best_rated_rhyme = max(rhyme_fellows, key=lambda item: item['rating'])
            # Small rating is not a rhyme.
            if best_rated_rhyme['rating'] < 0.5:
                scheme[line_idx] = next(letter_gen)
                relevant_comps[line_idx] = best_rated_rhyme['relevant_comps1']
                continue
            # todo take care of exceptions like AAAA->AABB
            scheme[line_idx] = scheme[best_rated_rhyme['index']]
            ratings[line_idx] = best_rated_rhyme['rating']
            relevant_comps[line_idx] = best_rated_rhyme['relevant_comps1']
            relevant_comps[best_rated_rhyme['index']] = best_rated_rhyme['relevant_comps2']
        stats.append({'scheme': scheme, 'ratings': ratings, 'relevant_components': relevant_comps})
    return stats


def adjust_matrix(matrixC, cons_idx, matrixV, vow_idx, stats):
    frequencies_C = [0]*len(cons_idx)
    frequencies_V = [0]*len(vow_idx)
    frequencies_CC = [[0]*len(cons_idx) for i in range(len(cons_idx))]
    frequencies_VV = [[0]*len(vow_idx) for i in range(len(vow_idx))]
    # Count frequencies for pairs and individual components.
    for song in stats:
        scheme = song['scheme']
        # Add individual frequencies of the first line.
        if song['relevant_components'][0]:
            for s in range(len(song['relevant_components'][0])):
                a = song['relevant_components'][0][s]
                if a in vow_idx:
                    ai = vow_idx.index(a)
                    frequencies_V[ai] += 1
                else:
                    ai = cons_idx.index(a)
                    frequencies_C[ai] += 1
        # Look at preceding lines for rhyme pairs.
        for i in range(1, len(song['relevant_components'])):
            if not song['relevant_components'][i]:
                continue
            # Add individual frequencies.
            for s in range(len(song['relevant_components'][i])):
                a = song['relevant_components'][i][s]
                if a in vow_idx:
                    ai = vow_idx.index(a)
                    frequencies_V[ai] += 1
                else:
                    ai = cons_idx.index(a)
                    frequencies_C[ai] += 1
            for lines_back in range(1, min(NO_OF_PRECEDING_LINES, i) + 1):
                if not song['relevant_components'][i-lines_back]:
                    continue
                # Add pair frequency if rhymes.
                if scheme[i] == scheme[i-lines_back]:
                    l = min(len(song['relevant_components'][i]), len(song['relevant_components'][i-lines_back]))
                    for s in range(1, l):
                        a, b = sorted([song['relevant_components'][i][-s],
                                       song['relevant_components'][i-lines_back][-s]])
                        if a in vow_idx and b in vow_idx:
                            ai = vow_idx.index(a)
                            bi = vow_idx.index(b)
                            frequencies_VV[ai][bi] += 1
                        else:
                            ai = cons_idx.index(a)
                            bi = cons_idx.index(b)
                            frequencies_CC[ai][bi] += 1
    # Create new matrices based on calculated frequencies.
    for i in range(len(matrixV)):
        for j in range(i+1, len(matrixV)):
            # T-score
            matrixV[i][j] = 0 if frequencies_VV[i][j] == 0 else (frequencies_VV[i][j] - frequencies_V[i]*frequencies_V[j]/sum(frequencies_V))/math.sqrt(frequencies_VV[i][j])
    for i in range(len(matrixC)):
        for j in range(i+1, len(matrixC)):
            # T-score
            matrixC[i][j] = 0 if frequencies_CC[i][j] == 0 else (frequencies_CC[i][j] - frequencies_C[i] * frequencies_C[j] / sum(frequencies_C)) / math.sqrt(frequencies_CC[i][j])
    return matrixC, matrixV


def save_matrix(matrix, ids, filename):
    with open(filename, 'w+') as output:
        output.write(','.join(x.ljust(5) for x in ids) + '\n')
        for i in range(len(matrix)):
            output.write(' '*i*6 + ",".join(format(x, "1.3f") for x in matrix[i][i:])+'\n')


def main():
    # split_dataset('../song_data/data/ENlyrics_final.json', 0.001, 0.001)
    data, cons_comp, vow_comp = extract_relevant_data('data/train_lyrics0.001.json')
    matrixC = initialize_matrix(len(cons_comp))
    matrixV = initialize_matrix(len(vow_comp))
    # matrixC, matrixV = first_step(data, matrixC, matrixV)
    n = 0
    while n < 1:
        stats = find_rhymes(data, (matrixC, cons_comp), (matrixV, vow_comp))
        matrixC, matrixV = adjust_matrix(matrixC, cons_comp, matrixV, vow_comp, stats)
        n += 1
    save_matrix(matrixC, cons_comp, 'data/matrixC.csv')
    save_matrix(matrixV, vow_comp, 'data/matrixV.csv')


if __name__=='__main__':
    main()
