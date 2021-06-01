import argparse
import json
import math
import random
import re
import sys

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
    def __init__(self, data_path, perfect_only, matrixC_path=None, matrixV_path=None):
        if data_path:
            self.data, self.cons_comp, self.vow_comp = self.extract_relevant_data(data_path)
        if perfect_only:
            self.cons_comp, self.matrixC = self._load_matrix('data/matrixC_identity.csv')
            self.vow_comp, self.matrixV = self._load_matrix('data/matrixV_identity.csv')
            self.init_value = 0
        elif matrixC_path and matrixV_path:
            self.cons_comp, self.matrixC = self._load_matrix(matrixC_path)
            self.vow_comp, self.matrixV = self._load_matrix(matrixV_path)
            self.init_value = 0.1
        else:
            self.matrixC = self._initialize_matrix(len(self.cons_comp))
            self.matrixV = self._initialize_matrix(len(self.vow_comp))
            self.init_value = 0.1
        self.freqC = [1/len(self.cons_comp)]*len(self.cons_comp)
        self.freqV = [1/len(self.vow_comp)]*len(self.vow_comp)
        # Default value to use when previously unseen component is found.
        self.perfect_only = perfect_only
        self.rhyme_rating_min = 0.8

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

    def _initialize_matrix(self, size, init_val=0.2):
        matrix = [[init_val]*size for i in range(size)]
        for i in range(len(matrix)):
            matrix[i][i] = 1.0
        return matrix

    # Get the phonemes after the last stress (the "relevant" part).
    def get_phonemes_after_last_stress(self, line):
        relevant = []
        idx = 0
        for i in range(len(line)-1, -1, -1):
            _, v, c2 = line[i]
            v = ' '.join(v)
            c2 = ' '.join(c2)
            if '1' in v or '2' in v:
                relevant = [v, c2]
                idx = i + 1
                break
        while idx < len(line):
            c1, v, c2 = line[idx]
            relevant.extend([' '.join(c1), ' '.join(v), ' '.join(c2)])
            idx += 1
        return relevant

    # Get components after the last stress for a pair - move stress if needed.
    def get_relevant_components_for_pair(self, first, second):
        stress_penalty = False
        last_stressed_idx1 = 4
        last_stressed_idx2 = 4
        # Find last stressed syllable index in both.
        for i in range(len(first)):
            _, v, _ = first[i]
            v = ''.join(v)
            if '1' in v or '2' in v:
                last_stressed_idx1 = i
        for i in range(len(second)):
            _, v, _ = second[i]
            v = ''.join(v)
            if '1' in v or '2' in v:
                last_stressed_idx2 = i
        # Select index of last stressed syllable (joined for both), add penalty if needed.
        if len(first) - last_stressed_idx1 != len(second) - last_stressed_idx2:
            stress_penalty = True
        relevent_last_sylls = max(len(first) - last_stressed_idx1, len(second) - last_stressed_idx2)
        if relevent_last_sylls > len(first):
            relevent_last_sylls = len(first)
        if relevent_last_sylls > len(second):
            relevent_last_sylls = len(second)
        # Separate the relevant components from given index.
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
        # Remove stress markings.
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
        new_row = [0]*idx + [self.init_value] * (len(matrix)- idx)
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
                matrix[i].insert(idx, self.init_value)
            elif i == idx:
                matrix[i].insert(idx, 1.0)
            else:
                matrix[i].insert(idx, 0)
        matrix[idx][idx] = 1.0

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
            # Best rhyme for each line with its parameters.
            rhymes = [{'rating': 0,
                       'rhyme_fellow': 0,
                       'relevant_components': None,
                       'relevant_components_rhyme_fellow': None,
                       'other_candidates': []} for i in range(len(song))]
            # Phonemes after last stress for each line (these are only used when no rhyme for the line is found).
            last_stressed_phonemes = [[]]*len(song)
            if song[0]:
                last_stressed_phonemes[0] = self.get_phonemes_after_last_stress(song[0][0])
            start_idx = 0
            for line_idx in range(start_idx + 1, len(song)):
                if not song[line_idx]:
                    continue
                last_stressed_phonemes[line_idx] = self.get_phonemes_after_last_stress(song[line_idx][0])
                # Find the best rated combination of pronunciation of this line and its selected predecessors.
                possible_rhymes = []
                for pronunciation1 in song[line_idx]:
                    # Look for rhyme fellow in preceding lines.
                    for lines_back in range(1, min(NO_OF_PRECEDING_LINES, line_idx) + 1):
                        if not song[line_idx-lines_back]:
                            continue
                        for pronunciation2 in song[line_idx-lines_back]:
                            rel_first, rel_second, stress_penalty = self.get_relevant_components_for_pair(pronunciation1, pronunciation2)
                            rating = self.get_rhyme_rating_simple_multiplication(rel_first, rel_second, stress_penalty)
                            result = {'rating': rating,
                                      'relevant_comps': rel_first,
                                      'relevant_comps_fellow': rel_second,
                                      'relative_index': lines_back}
                            possible_rhymes.append(result)
                if not possible_rhymes:
                    continue
                # Select the rhyme fellow with best rating.
                best_rated_rhyme = max(possible_rhymes, key=lambda item: item['rating'])
                # Small rating is not a rhyme.
                if best_rated_rhyme['rating'] > self.rhyme_rating_min:
                    rhymes[line_idx]['rating'] = best_rated_rhyme['rating']
                    rhymes[line_idx]['rhyme_fellow'] = - best_rated_rhyme['relative_index']
                    rhymes[line_idx]['relevant_components_rhyme_fellow'] = best_rated_rhyme['relevant_comps_fellow']
                    # Keep other candidates in case of pronunciation conflict.
                    rhymes[line_idx]['other_candidates'] = list(filter(lambda c: c['rating'] > self.rhyme_rating_min and c != best_rated_rhyme, possible_rhymes))
                rhymes[line_idx]['relevant_components'] = best_rated_rhyme['relevant_comps']
            stats.append(self._revise_and_create_scheme(rhymes, last_stressed_phonemes))
        return stats

    # Create rhyme scheme.
    # todo avoid pronunciation conflicts.
    def _revise_and_create_scheme(self, rhymes, relevant_parts):
        # Lists of line index with the same rhyme.
        rhyme_groups = []
        for i in range(len(rhymes)):
            if not relevant_parts[i]:
                continue
            if rhymes[i]['rhyme_fellow'] == 0:
                rhyme_groups.append([i])
            else:
                for group in rhyme_groups:
                    if i+rhymes[i]['rhyme_fellow'] in group:
                        # Keep the pronunciation used for the rhyme.
                        relevant_parts[i] = rhymes[i]['relevant_components']
                        if relevant_parts[i+rhymes[i]['rhyme_fellow']] != rhymes[i]['relevant_components_rhyme_fellow']:
                            print(f"CONFLICT FOUND:{i}")
                        relevant_parts[i + rhymes[i]['rhyme_fellow']] = rhymes[i]['relevant_components_rhyme_fellow']
                        group.append(i)
                        continue
        # Take care of exceptions like AAAA->AABB
        revised_groups = []
        for group in rhyme_groups:
            if len(group) < 4:
                revised_groups.append(group)
                continue
            i = 0
            separated_indexes = []
            while i < len(group)-4:
                # If we have four consecutive lines and ratings are stronger between first two and second two -> pair them.
                if group[i+1] == group[i]+1 and \
                    group[i+2] == group[i]+2 and \
                    group[i+3] == group[i]+3 and \
                    rhymes[i+3]['rating'] > rhymes[i+2]['rating'] and \
                    rhymes[i+1]['rhyme_fellow'] == -1 and \
                    rhymes[i+3]['rhyme_fellow'] == -1:
                        revised_groups.append([group[i], group[i+1]])
                        separated_indexes.extend([group[i], group[i+1]])
                        # Remove the weaker rhyme.
                        rhymes[i+2]['rating'] = 0
                        rhymes[i+2]['rhyme_fellow'] = 0
                # Always jump 2 to avoid separating one rhyming line from the rest.
                i += 2
            # Remove indexes that separated.
            for idx in separated_indexes:
                group.remove(idx)
            # Make sure that no spaces are further than 3 lines.
            leftover_idx = 0
            for i in range(len(group)-1):
                if group[i+1] - group[i] > 3:
                    revised_groups.append(group[:i])
                    leftover_idx = i+1
            revised_groups.append(group[leftover_idx:])
        # Assign rhyme scheme letters.
        letter_gen = next_letter_generator()
        scheme = ['']*len(rhymes)
        for i in range(len(rhymes)):
            for group in revised_groups:
                # Assign one letter to the entire rhyme group.
                if i in group:
                    l = next(letter_gen)
                    for idx in group:
                        scheme[idx] = l
                    revised_groups.remove(group)
                    continue
        stats = {'scheme': scheme,
                 'ratings': rhymes,
                 'relevant_components': relevant_parts}
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
        rel_freqC = [(f + len(self.matrixC))/totalC for f in frequencies_C]
        totalV = sum(frequencies_V)
        rel_freqV = [(f + len(self.matrixV)) / totalV for f in frequencies_V]
        # Create new matrices based on calculated frequencies.
        totalVV = sum(map(sum, frequencies_VV))
        totalCC = sum(map(sum, frequencies_CC))
        for i in range(len(self.matrixV)):
            for j in range(i+1, len(self.matrixV)):
                rel_freq = (frequencies_VV[i][j]+1)/totalVV
                self.matrixV[i][j] = rel_freq/(rel_freq + rel_freqV[i]*rel_freqV[j])
        for i in range(len(self.matrixC)):
            for j in range(i+1, len(self.matrixC)):
                rel_freq = (frequencies_CC[i][j]+1)/totalCC
                self.matrixC[i][j] = rel_freq/(rel_freq + rel_freqC[i]*rel_freqC[j])
        self.freqC = frequencies_C
        self.freqV = frequencies_V
        self._print_state()
        return

    def save_matrixC(self, path):
        self._save_matrix(self.matrixC, self.cons_comp, path)

    def save_matrixV(self, path):
        self._save_matrix(self.matrixV, self.vow_comp, path)

    # Prints current state of matrices and component frequencies.
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
        with open(filename) as matrix_input:
            idxs = [id.strip() for id in matrix_input.readline().split(',')]
            size = len(idxs)
            matrix = [[0] * size for i in range(size)]
            for i in range(size):
                line = [float(item.strip()) for item in matrix_input.readline().split(',')]
                for j in range(i, size):
                    matrix[i][j] = line[j-i]
        return idxs, matrix


def main(args):
    if args.do_train:
        if not args.matrix_C_file and not args.matrix_V_file:
            if args.ratio:
                # Prepare dataset and initialize the detector.
                split_dataset(args.train_file, args.ratio, args.ratio)
                detector = RhymeDetector('data/train_lyrics'+str(args.ratio)+'.json', args.perfect_only)
                args.test_file = 'data/test_lyrics' + str(args.ratio) + '.json'
            else:
                # Initialize the detector from train file.
                detector = RhymeDetector(args.train_file, args.perfect_only)
            detector.save_matrixC('data/matrixC_init.csv')
            detector.save_matrixV('data/matrixV_init.csv')
        else:
            detector = RhymeDetector(args.train_file, args.perfect_only, args.matrix_C_file, args.matrix_V_file)
        detector.save_matrixC('data/matrixC_init.csv')
        detector.save_matrixV('data/matrixV_init.csv')
        n = 0
        # Train the detector.
        while n < 3:
            print(f"ITERATION {n+1}")
            stats = detector.find_rhymes()
            detector.adjust_matrix(stats)
            detector.save_matrixC('data/matrixC_iter'+str(n)+'.csv')
            detector.save_matrixV('data/matrixV_iter'+str(n)+'.csv')
            n += 1
    if args.do_test:
        # Test the detector.
        if not args.do_train:
            matrixC_file = args.matrix_C_file
            matrixV_file = args.matrix_V_file
            detector = RhymeDetector(None, args.perfect_only, matrixC_file, matrixV_file)
        test_data_pron, cons, vows = detector.extract_relevant_data(args.test_file)
        with open(args.test_file) as input:
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
                print(f"{stats[s]['scheme'][l]:<2}",
                      f"{stats[s]['ratings'][l]['rating']:5.3f}" if isinstance(stats[s]['ratings'][l]['rating'], float) else f"{stats[s]['ratings'][l]['rating']:<5}",
                      f"{stats[s]['ratings'][l]['rhyme_fellow']:<3}",
                      f"{stats[s]['relevant_components'][l]}", test_data[s]['lyrics'][l])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_test',  default=False, action='store_true')
    parser.add_argument('--do_train',  default=False, action='store_true')
    parser.add_argument('--train_file', required='do_train' in sys.argv, help="Either a file for train file generation (if ratio is given) or already generated train file.")
    parser.add_argument('--test_file', required='do_test' in sys.argv and '--ratio' not in sys.argv)
    parser.add_argument('--ratio', type=float, help="If given, data for training/testing are generated from given train_file in given ratio.")
    parser.add_argument('--matrix_C_file', help="Matrix loaded for testing. If training selected, this matrix will be loaded as initialization matrix.")
    parser.add_argument('--matrix_V_file', help="Matrix loaded for testing. If training selected, this matrix will be loaded as initialization matrix.")
    parser.add_argument('--perfect_only', default=False, action='store_true')

    args = parser.parse_args([#'--train_file', '../song_data/data/ENlyrics_final.json',
                              # '--train_file', 'data/train_lyrics0.001.json',
                              '--test_file', 'data/test_lyrics0.001.json',
                              # '--ratio', '0.001',
                              '--matrix_C_file', 'data/matrixC_identity.csv',
                              '--matrix_V_file', 'data/matrixV_identity.csv',
                              # '--do_train',
                              '--do_test',
                              '--perfect_only'])
    main(args)
