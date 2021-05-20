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


class RhymeDetector:
    def __init__(self, data_path = None, matrixC_path = None, matrixV_path = None):
        if data_path:
            self.data, self.cons_comp, self.vow_comp = self.extract_relevant_data(data_path)
            self.matrixC = self._initialize_matrix(len(self.cons_comp))
            self.matrixV = self._initialize_matrix(len(self.vow_comp))
        if matrixC_path and matrixV_path:
            self.cons_comp, self.matrixC = self._load_matrix(matrixC_path)
            self.vow_comp, self.matrixV = self._load_matrix(matrixV_path)
        self.freqC = [1/len(self.cons_comp)]*len(self.cons_comp)
        self.freqV = [1/len(self.vow_comp)]*len(self.vow_comp)
        # Deafult value to use when previously unseen component is found.
        self.new_value = 0.1

    # Returns an array of songs.
    # Each song is an array of lines.
    # Each line is an array possible pronunciations.
    # Each pronunciation is an array of 4 last syllables.
    # Each syllable is a triplet CVC.
    def extract_relevant_data(self, dataset_file):
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

    def _initialize_matrix(self, size):
        matrix = [[0.6]*size for i in range(size)]
        for i in range(len(matrix)):
            matrix[i][i] = 0.99
        return matrix

    # Get components after the last stress.
    def get_relevant_components(self, first, second):
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

    # Tried to work with previously unseen component - add it to the matrix.
    def add_new_component(self, matrix, ids, a):
        idx = 0
        while idx < len(ids) and ids[idx] < a:
            idx += 1
        ids.insert(idx, a)
        # Add a row.
        new_row = [0]*idx + [self.new_value] * (len(matrix)- idx)
        if idx == len(matrix):
            matrix.append(new_row)
        else:
            to_shift = matrix[idx:]
            matrix[idx] = new_row
            for i in range(idx+1, len(matrix)):
                matrix[i] = to_shift[i-idx-1]
            matrix.append(to_shift[-1])
        # Add column.
        for i in range(len(matrix)):
            if i < idx:
                matrix[i].insert(idx, self.new_value)
            elif i == idx:
                matrix[i].insert(idx, 0.99)
            else:
                matrix[i].insert(idx, 0)
        matrix[idx][idx] = 0.99

    # Get rhyme rating using multiplication of partial probabilities with inverse probabilities.
    def get_rhyme_rating(self, first, second, stress_penalty):
        product_prob = 1
        product_inverse_prob = 1
        vowel = True
        next_vowel = False
        for i in range(len(first)):
            a, b = sorted([first[i], second[i]])
            if vowel:
                # If new component found, add it to the matrix.
                if a not in self.vow_comp:
                    self.add_new_component(self.matrixV, self.vow_comp, a)
                if b not in self.vow_comp:
                    self.add_new_component(self.matrixV, self.vow_comp, b)
                # Get the index a multiply with given probability.
                ai = self.vow_comp.index(a)
                bi = self.vow_comp.index(b)
                product_prob *= self.matrixV[ai][bi]
                product_inverse_prob *= 1-self.matrixV[ai][bi]
            else:
                # If new component found, add it to the matrix.
                if a not in self.cons_comp:
                    self.add_new_component(self.matrixC, self.cons_comp, a)
                if b not in self.cons_comp:
                    self.add_new_component(self.matrixC, self.cons_comp, b)
                ai = self.cons_comp.index(a)
                bi = self.cons_comp.index(b)
                product_prob *= self.matrixC[ai][bi]
                product_inverse_prob *= 1 - self.matrixC[ai][bi]
            last_vowel = vowel
            vowel = next_vowel
            if vowel:
                next_vowel = False
            elif not last_vowel:
                next_vowel = True
        rating = product_prob/(product_prob + product_inverse_prob)
        if stress_penalty:
            rating -= 0.1
        return rating

    def get_rhyme_rating_simple_multiplication(self, first, second, stress_penalty):
        rating = 1
        vowel = True
        next_vowel = False
        for i in range(len(first)):
            a, b = sorted([first[i], second[i]])
            if vowel:
                ai = self.vow_comp.index(a)
                bi = self.vow_comp.index(b)
                rating *= self.matrixV[ai][bi]
            else:
                ai = self.cons_comp.index(a)
                bi = self.cons_comp.index(b)
                rating *= self.matrixC[ai][bi]
            last_vowel = vowel
            vowel = next_vowel
            if vowel:
                next_vowel = False
            elif not last_vowel:
                next_vowel = True
        if stress_penalty:
            rating -= 0.1
        return rating

    # Returns a list of song objects.
    # Song object contains: scheme, ratings, relevant_components.
    def find_rhymes(self, data=None):
        stats = []
        if not data:
            data = self.data
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
                    scheme[line_idx] = NOT_AVAILABLE
                    continue
                for pronunciation1 in song[line_idx]:
                    for lines_back in range(1, min(NO_OF_PRECEDING_LINES, line_idx) + 1):
                        for pronunciation2 in song[line_idx-lines_back]:
                            rel_first, rel_second, stress_penalty = self.get_relevant_components(pronunciation1, pronunciation2)
                            rating = self.get_rhyme_rating(rel_first, rel_second, stress_penalty)
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
                if best_rated_rhyme['rating'] < 0.8:
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

    def adjust_matrix(self, stats):
        frequencies_C = [0]*len(self.cons_comp)
        frequencies_V = [0]*len(self.vow_comp)
        frequencies_CC = [[0]*len(self.cons_comp) for i in range(len(self.cons_comp))]
        frequencies_VV = [[0]*len(self.vow_comp) for i in range(len(self.vow_comp))]
        # Count frequencies for pairs and individual components.
        for song in stats:
            scheme = song['scheme']
            # Add individual frequencies of the first line.
            if song['relevant_components'][0]:
                for s in range(len(song['relevant_components'][0])):
                    a = song['relevant_components'][0][s]
                    if a in self.vow_comp:
                        ai = self.vow_comp.index(a)
                        frequencies_V[ai] += 1
                    else:
                        ai = self.cons_comp.index(a)
                        frequencies_C[ai] += 1
            # Look at preceding lines for rhyme pairs.
            for i in range(1, len(song['relevant_components'])):
                if not song['relevant_components'][i]:
                    continue
                # Add individual frequencies.
                for s in range(len(song['relevant_components'][i])):
                    a = song['relevant_components'][i][s]
                    if a in self.vow_comp:
                        ai = self.vow_comp.index(a)
                        frequencies_V[ai] += 1
                    else:
                        ai = self.cons_comp.index(a)
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
                            if a in self.vow_comp and b in self.vow_comp:
                                ai = self.vow_comp.index(a)
                                bi = self.vow_comp.index(b)
                                frequencies_VV[ai][bi] += 1
                            else:
                                ai = self.cons_comp.index(a)
                                bi = self.cons_comp.index(b)
                                frequencies_CC[ai][bi] += 1
        # Calculate relative frequencies.
        totalC = sum(frequencies_C)
        rel_freqC = [f/totalC for f in frequencies_C]
        totalV = sum(frequencies_V)
        rel_freqV = [f / totalV for f in frequencies_V]
        # Create new matrices based on calculated frequencies.
        totalVV = sum(map(sum, frequencies_VV))
        totalCC = sum(map(sum, frequencies_CC))
        for i in range(len(self.matrixV)):
            for j in range(i+1, len(self.matrixV)):
                if frequencies_VV[i][j] == 0:
                    self.matrixV[i][j] = 0.0001
                else:
                    rel_freq = frequencies_VV[i][j]/totalVV
                    self.matrixV[i][j] = rel_freq/(rel_freq + rel_freqV[i]*rel_freqV[j])
        for i in range(len(self.matrixC)):
            for j in range(i+1, len(self.matrixC)):
                if frequencies_CC[i][j] == 0:
                    self.matrixC[i][j] = 0.0001
                else:
                    rel_freq = frequencies_CC[i][j]/totalCC
                    self.matrixC[i][j] = rel_freq/(rel_freq + rel_freqC[i]*rel_freqC[j])
        self.freqC = frequencies_C
        self.freqV = frequencies_V
        self._print_state()
        return

    def save_matrixC(self, path):
        self._save_matrix(self.matrixC, self.cons_comp, path)

    def save_matrixV(self, path):
        self._save_matrix(self.matrixV, self.vow_comp, path)

    def _print_state(self):
        print('CONSONANT MATRIX')
        self._print_matrix(self.matrixC, self.cons_comp)
        print('VOWEL MATRIX')
        self._print_matrix(self.matrixV, self.vow_comp)
        print('CONSONANT FREQUENCIES')
        for i in range(len(self.cons_comp)):
            print(f'{self.cons_comp[i]}: {self.freqC[i]}')
        print('VOWEL FREQUENCIES')
        for i in range(len(self.vow_comp)):
            print(f'{self.vow_comp[i]}: {self.freqV[i]}')

    def _print_matrix(self, matrix, ids):
        print(','.join(x.ljust(5) for x in ids))
        for i in range(len(matrix)):
            print(' ' * i * 6 + ",".join(format(x, "1.3f") for x in matrix[i][i:]))

    def _save_matrix(self, matrix, ids, filename):
        with open(filename, 'w+') as output:
            output.write(','.join(x.ljust(7) for x in ids) + '\n')
            for i in range(len(matrix)):
                output.write(' '*i*8 + ",".join(format(x, "1.5f") for x in matrix[i][i:])+'\n')

    def _load_matrix(self, filename):
        with open(filename) as input:
            idxs = [id.strip() for id in input.readline().split(',')]
            size = len(idxs)
            matrix = [[0] * size for i in range(size)]
            for i in range(size):
                line = [float(item.strip()) for item in input.readline().split(',')]
                for j in range(i, size):
                    matrix[i][j] = line[j-i]
        return idxs, matrix


def main():
    # Prepare dataset and initialize the detector.
    # split_dataset('../song_data/data/ENlyrics_final.json', 0.001, 0.001)
    # detector = RhymeDetector('data/train_lyrics0.001.json')
    # n = 0
    # # Train the detector.
    # while n < 10:
    #     print(f"ITERATION {n+1}")
    #     stats = detector.find_rhymes()
    #     detector.adjust_matrix(stats)
    #     n += 1
    # detector.save_matrixC('data/matrixC.csv')
    # detector.save_matrixV('data/matrixV.csv')
    # Test the detector.
    detector = RhymeDetector(None, 'data/matrixC.csv', 'data/matrixV.csv')
    test_data_file = 'data/test_lyrics0.001.json'
    test_data_pron, cons, vows = detector.extract_relevant_data(test_data_file)
    with open(test_data_file) as input:
        test_data = json.load(input)
    # Add new components.
    for c in cons:
        if c not in detector.cons_comp:
            detector.add_new_component(detector.matrixC, detector.cons_comp, c)
    for v in vows:
        if v not in detector.vow_comp:
            detector.add_new_component(detector.matrixV, detector.vow_comp, v)
    stats = detector.find_rhymes(test_data_pron)
    for s in range(len(test_data)):
        print(f"NEXT SONG: {test_data[s]['title']}")
        for l in range(len(test_data[s]['lyrics'])):
            print(f"{stats[s]['scheme'][l]:<2}", f"{stats[s]['ratings'][l]:<3}",
                  f"{stats[s]['relevant_components'][l]}", test_data[s]['lyrics'][l])


if __name__=='__main__':
    main()
    # detector = RhymeDetector(None, 'data/matrixC.csv', 'data/matrixV.csv')
    # detector.add_new_component(detector.matrixV, detector.vow_comp, 'AAA')