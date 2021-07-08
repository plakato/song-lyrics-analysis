import argparse
import copy
import json
import math
import pickle
import random
import re
import sys

import sklearn

from constants import NO_OF_PRECEDING_LINES
from rhyme_detector_v1 import get_pronunciations_for_n_syllables, next_letter_generator


class RhymeDetector:
    def __init__(self, perfect_only=False, matrix_path=None, zero_value=0.001, rhyme_rating_min=0.8, window=3, verbose=False):
        self.data = []
        self.verbose = verbose
        self.separator = '&'
        self.oscilation_check = dict()
        global NO_OF_PRECEDING_LINES
        NO_OF_PRECEDING_LINES = window
        # Value assigned when the component is not in the matrix (cooc).
        self.zero_value = zero_value
        # Initialization value for the matrix components at the beginning of training.
        self.init_value = 0.2
        self.perfect_only = perfect_only
        # Minimal rating for a pair of lines to be judged a rhyme.
        self.rhyme_rating_min = rhyme_rating_min
        # Default character used
        # - when rating is not available (first line in rhyme group)
        # - when the line doesn't rhyme (in place of scheme letter)
        self.non_rhyme_char = "-"
        self.cons_comp = set()
        self.vow_comp = set()
        if perfect_only:
            self.cooc = dict()
            self.rhyme_rating_min = 1.0
        elif matrix_path:
            self.cooc = self._load_matrix(matrix_path)
        else:
            self._initialize_matrix()

    # Returns an array of songs.
    # Each song is an array of lines.
    # Each line is an array possible pronunciations.
    # Each pronunciation is an array of 4 last syllables.
    # Each syllable is a triplet CVC.
    def load_and_preprocess_data_from_file(self, dataset_file):
        self.cons_comp = set()
        self.vow_comp = set()
        # In new data, every song is an array of possible pronunciations of last 4 syllables.
        new_data = []
        with open(dataset_file) as file:
            data = json.load(file)
            for song in data:
                lyrics_last4 = self.preprocess_lyrics(song['lyrics'])
                new_data.append(lyrics_last4)
        self.cons_comp = sorted(list(self.cons_comp))
        self.vow_comp = sorted(list(self.vow_comp))
        self.data = new_data
        return new_data

    # Preprocess lyrics for one song - extract pronunciation of last 4 syllables.
    def preprocess_lyrics(self, lyrics):
        lyrics_last4 = []
        for line in lyrics:
            if line == '':
                lyrics_last4.append([])
                continue
            prons = get_pronunciations_for_n_syllables(line, 4)
            # Add all components (without stress) to sets of unique components.
            for pron in prons:
                for syll in pron:
                    c1, v, c2 = syll
                    self.cons_comp.add(re.sub(r'\d+', '', ' '.join(c1).strip()))
                    self.cons_comp.add(re.sub(r'\d+', '', ' '.join(c2).strip()))
                    self.vow_comp.add(re.sub(r'\d+', '', ' '.join(v).strip()))
            lyrics_last4.append(prons)
        return lyrics_last4

    # Initialize with self.init_value all vowel combinations and all consonant combinations.
    def _initialize_matrix(self):
        self.cooc = dict()
        for i in range(len(self.cons_comp)):
            for j in range(i+1, len(self.cons_comp)):
                key = self.separator.join(sorted([self.cons_comp[i], self.cons_comp[j]]))
                self.cooc[key] = self.init_value
        for i in range(len(self.vow_comp)):
            for j in range(i + 1, len(self.vow_comp)):
                key = self.separator.join(sorted([self.vow_comp[i], self.vow_comp[j]]))
                self.cooc[key] = self.init_value

    # Complete full analysis of one song's lyrics.
    def analyze_lyrics(self, lyrics):
        self.data = [self.preprocess_lyrics(lyrics)]
        stats = self.find_rhymes()
        return stats[0]

    # Get the phonemes after the last stress (the "relevant" part).
    # Line is a list of triplets CVC.
    @staticmethod
    def get_phonemes_after_last_stress(line):
        relevant = []
        idx = 0
        for i in range(len(line)-1, -1, -1):
            _, v, c2 = line[i]
            v = ' '.join(v)
            c2 = ' '.join(c2)
            if '1' in v or '2' in v:
                v = re.sub(r"[12]", '', v)
                relevant = [v, c2]
                idx = i + 1
                break
        while idx < len(line):
            c1, v, c2 = line[idx]
            v = re.sub(r"[012]", '', ' '.join(v))
            relevant.extend([' '.join(c1), v, ' '.join(c2)])
            idx += 1
        return relevant

    # Get components after the last stress for a pair - move stress if needed.
    @staticmethod
    def get_relevant_components_for_pair(first, second):
        stress_penalty = False
        last_stressed_idx1 = 4
        last_stressed_idx2 = 4
        # Solve for the case when one is empty.
        if not first:
            return [], RhymeDetector.get_phonemes_after_last_stress(second), True
        if not second:
            return RhymeDetector.get_phonemes_after_last_stress(second), [], True
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
        relevent_last_sylls = min(len(first) - last_stressed_idx1, len(second) - last_stressed_idx2)
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

    # Get rhyme rating using multiplication of partial probabilities with inverse probabilities.
    def get_rhyme_rating(self, first, second, stress_penalty):
        product_prob = 1
        product_inverse_prob = 1
        for i in range(len(first)):
            a, b = sorted([first[i], second[i]])
            if a == b:
                product_inverse_prob *= self.zero_value
                continue
            index = self.separator.join([a, b])
            val = self.cooc.get(index, self.zero_value)
            product_prob *= val
            product_inverse_prob *= 1-val
        rating = product_prob/(product_prob + product_inverse_prob)
        if stress_penalty:
            rating -= 0.1
        return rating

    def get_rhyme_rating_simple_multiplication(self, first, second, stress_penalty):
        # Empty lines can't rhyme.
        if first == [] or second == []:
            return 0
        rating = 1
        for i in range(len(first)):
            a, b = sorted([first[i], second[i]])
            if a == b:
                continue
            index = self.separator.join([a, b])
            rating *= self.cooc.get(index, self.zero_value)
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
            rhymes = [{'rating': None,
                       'rhyme_fellow': 0,
                       'relevant_components': None,
                       'relevant_components_rhyme_fellow': None,
                       'stress_moved': False,
                       'other_candidates': []} for i in range(len(song))]
            # Phonemes after last stress for each line (these are only used when no rhyme for the line is found).
            last_stressed_phonemes = [[]]*len(song)
            if song[0]:
                last_stressed_phonemes[0] = self.get_phonemes_after_last_stress(song[0][0])
            else:
                rhymes[0]['rating'] = self.non_rhyme_char
            start_idx = 0
            for line_idx in range(start_idx + 1, len(song)):
                if not song[line_idx]:
                    rhymes[line_idx]['rating'] = self.non_rhyme_char
                    continue
                last_stressed_phonemes[line_idx] = self.get_phonemes_after_last_stress(song[line_idx][0])
                # Find the best rated combination of pronunciation of this line and its selected predecessors.
                possible_rhymes = []
                for pronunciation1 in song[line_idx]:
                    # Look for rhyme fellow in preceding lines.
                    lines_back = 0
                    lines_evaluated = 0
                    while lines_evaluated < NO_OF_PRECEDING_LINES and line_idx - lines_back > 0:
                        lines_back += 1
                        if not song[line_idx-lines_back]:
                            lines_evaluated += 1
                            continue
                        lines_evaluated += 1
                        for pronunciation2 in song[line_idx-lines_back]:
                            rel_first, rel_second, stress_penalty = self.get_relevant_components_for_pair(pronunciation1, pronunciation2)
                            while len(rel_first) >= 2 and len(rel_second) >= 2:
                                rating = self.get_rhyme_rating_simple_multiplication(rel_first, rel_second, stress_penalty)
                                if rating >= self.rhyme_rating_min:
                                    result = {'rating': rating,
                                              'relevant_components': rel_first,
                                              'relevant_components_rhyme_fellow': rel_second,
                                              'rhyme_fellow': - lines_back,
                                              'stress_moved': stress_penalty}
                                    if result not in possible_rhymes:
                                        possible_rhymes.append(result)
                                # Try truncating the relevant part and look for rhyme only closer to the end.
                                rel_first = rel_first[3:] if len(rel_first) > 2 else []
                                rel_second = rel_second[3:] if len(rel_second) > 2 else []
                                stress_penalty = True
                if not possible_rhymes:
                    continue
                # Select the rhyme fellow with best rating.
                best_rated_rhyme = max(possible_rhymes, key=lambda item: item['rating'])
                rhymes[line_idx] = best_rated_rhyme
                # Keep other candidates in case of pronunciation conflict (sorted - descending rating).
                rhymes[line_idx]['other_candidates'] = sorted(list(filter(lambda c: c != best_rated_rhyme, possible_rhymes)), key=lambda item: item['rating'], reverse=True)
            song_stats = self._revise_and_create_scheme(rhymes, last_stressed_phonemes)
            song_stats['song_rating'] = self.song_rating(rhymes)
            stats.append(song_stats)
        return stats

    # Create rhyme scheme - cluster rhymes into groups, yield higher ratings, assign letters.
    def _revise_and_create_scheme(self, rhymes, relevant_parts):
        # Lists of line index with the same rhyme.
        rhyme_groups = []
        for i in range(len(rhymes)):
            if not relevant_parts[i]:
                continue
            if rhymes[i]['rhyme_fellow'] == 0:
                rhyme_groups.append([i])
                continue
            for group in rhyme_groups:
                if i+rhymes[i]['rhyme_fellow'] in group:
                    if len(group) == 1:
                        relevant_parts[i + rhymes[i]['rhyme_fellow']] = rhymes[i]['relevant_components_rhyme_fellow']
                        group.append(i)
                    # If conflict found, try other candidates or forget about the rhyme and create new group.
                    elif relevant_parts[i+rhymes[i]['rhyme_fellow']] != rhymes[i]['relevant_components_rhyme_fellow']:
                        better_cand_found = False
                        for cand in rhymes[i]['other_candidates']:
                            if cand['relevant_components_rhyme_fellow'] == relevant_parts[i+cand['rhyme_fellow']] and i + cand['rhyme_fellow'] in group:
                                rhymes[i] = cand
                                better_cand_found = True
                                group.append(i)
                                break
                        if not better_cand_found:
                            rhymes[i]['rating'] = None
                            rhymes[i]['rhyme_fellow'] = 0
                            rhyme_groups.append([i])
                    else:
                        group.append(i)
                    # Keep the pronunciation used for the rhyme.
                    relevant_parts[i] = rhymes[i]['relevant_components']
                    break
        revised_groups, rhymes = self.solve_exceptions_in_rhymes(rhyme_groups, rhymes)
        scheme = self.assign_scheme_letters(rhymes, revised_groups)
        stats = {'scheme': scheme, 'ratings': rhymes, 'relevant_components': relevant_parts}
        return stats

    def solve_exceptions_in_rhymes(self, rhyme_groups, rhymes):
        # rhyme_groups = [[0,1], [2,3], ...]
        # rhymes = [{'rating': 0.9, 'rhyme_fellow': -2}, {'rating': 0.8, 'rhyme_fellow': -1},...]
        # Take care of exceptions where splitting rhyme group would yield better song rating.
        revised_groups = []
        while len(rhyme_groups) > 0:
            group = rhyme_groups.pop()
            # Remove small groups.
            if len(group) < 4:
                revised_groups.append(group)
                continue
            # Extract imperfect rhymes (to consider removing them).
            imperfects = [i for i in group if rhymes[i]['rating'] and 0 < rhymes[i]['rating'] < 1]
            imperfects = sorted(imperfects, key=lambda imp: rhymes[imp]['rating'])
            # Try removing rhyme - if the group splits and achieves higher rating, keep it.
            removed = False
            old_rating = self.song_rating(rhymes)
            for imperfect in imperfects:
                # Will both lines still be a part of some rhyme? Otherwise higher rating is impossible.
                fellow_idx = imperfect + rhymes[imperfect]['rhyme_fellow']
                without_imperfect = copy.deepcopy(group)
                without_imperfect.remove(imperfect)
                without_imperfect.remove(fellow_idx)
                if any(i+rhymes[i]['rhyme_fellow'] == imperfect for i in group) and \
                    (any(i+rhymes[i]['rhyme_fellow'] == fellow_idx for i in without_imperfect) or rhymes[fellow_idx]['rating']):
                    removed_imperfect = copy.deepcopy(rhymes)
                    removed_imperfect[imperfect]['rating'] = None
                    removed_imperfect[imperfect]['rhyme_fellow'] = 0
                    new_rating = self.song_rating(removed_imperfect)
                    if new_rating > old_rating:
                        rhymes = removed_imperfect
                        new_groups = self._get_new_groups(group, rhymes)
                        rhyme_groups.extend(new_groups)
                        removed = True
                        break
            if not removed:
                revised_groups.append(group)
        return revised_groups, rhymes

    def _get_new_groups(self, group, rhymes):
        group.sort()
        new_groups = [[group.pop(0)]]
        for elem in group:
            group_found = False
            for g in new_groups:
                if elem+rhymes[elem]['rhyme_fellow'] in g:
                    g.append(elem)
                    group_found = True
                    break
            if not group_found:
                new_groups.append([elem])
        # Make sure that no spaces are further than n lines.
        no_space_groups = []
        for group in new_groups:
            leftover_idx = 0
            for i in range(len(group) - 1):
                if group[i + 1] - group[i] > NO_OF_PRECEDING_LINES:
                    no_space_groups.append(group[leftover_idx:i+1])
                    leftover_idx = i + 1
            no_space_groups.append(group[leftover_idx:])
        return no_space_groups

    def assign_scheme_letters(self, rhymes, revised_groups):
        # Assign rhyme scheme letters.
        letter_gen = next_letter_generator()
        scheme = [self.non_rhyme_char]*len(rhymes)
        # For each line find and deal with its group.
        for i in range(len(rhymes)):
            for group in revised_groups:
                first = min(group)
                if i in group:
                    # Assign neutral character for non-rhymes.
                    if len(group) == 1:
                        scheme[i] = self.non_rhyme_char
                        rhymes[i]['rating'] = 0
                    else:
                        # Assign one letter to the entire rhyme group.
                        l = next(letter_gen)
                        for idx in group:
                            # First rhyme fellow doesn't have a rating.
                            if idx == first:
                                rhymes[idx]['rating'] = self.non_rhyme_char
                            scheme[idx] = l
                    revised_groups.remove(group)
                    break
        return scheme

    def adjust_matrix(self, stats):
        changed = True
        frequencies_indiv = dict()
        frequencies_pairs = dict()
        # Count frequencies for pairs and individual components.
        for song in stats:
            scheme = song['scheme']
            # Look at preceding lines for rhyme pairs.
            for i in range(1, len(song['relevant_components'])):
                if not song['relevant_components'][i]:
                    continue
                for lines_back in range(1, min(NO_OF_PRECEDING_LINES, i) + 1):
                    if not song['relevant_components'][i-lines_back]:
                        continue
                    # Add pair frequency if rhymes.
                    if scheme[i] == scheme[i-lines_back]:
                        l = min(len(song['relevant_components'][i]), len(song['relevant_components'][i-lines_back]))
                        for s in range(1, l):
                            a, b = sorted([song['relevant_components'][i][-s],
                                           song['relevant_components'][i-lines_back][-s]])
                            if a == b:
                                continue
                            index = self.separator.join([a, b])
                            if index in frequencies_pairs:
                                frequencies_pairs[index] += 1
                            else:
                                frequencies_pairs[index] = 1
                            if a in frequencies_indiv:
                                frequencies_indiv[a] += 1
                            else:
                                frequencies_indiv[a] = 1
                            if b in frequencies_indiv:
                                frequencies_indiv[b] += 1
                            else:
                                frequencies_indiv[b] = 1
        new_cooc = self.calculate_new_matrix_from_frequencies(frequencies_indiv, frequencies_pairs)
        if self.cooc == new_cooc or new_cooc == self.oscilation_check:
            changed = False
        self.oscilation_check = self.cooc
        self.cooc = new_cooc
        if self.verbose:
            self._print_state()
        return changed

    def song_rating(self, rhymes):
        summed = 0
        n = 0
        for r in rhymes:
            if type(r['rating']) == int or type(r['rating']) == float:
                n += 1
                summed += r['rating']
        return summed/n

    def calculate_new_matrix_from_frequencies(self, frequencies_indiv, frequencies_pairs):
        # Calculate relative frequencies for individual keys that occur in rhyme pairs => perfect match doesn't count
        total_indiv = sum(frequencies_indiv.values())
        rel_freq_indiv = dict()
        for key in frequencies_indiv:
            rel_freq_indiv[key] = (frequencies_indiv[key] + len(frequencies_indiv)) / total_indiv
        # Create new matrices based on calculated frequencies.
        total_pairs = sum(frequencies_pairs.values())
        new_cooc = dict.fromkeys(frequencies_pairs.keys(), 0)
        for key in frequencies_pairs:
            key_elem = key.split(self.separator)
            rel_freq = (frequencies_pairs[key] + 1) / total_pairs
            new_cooc[key] = rel_freq / (rel_freq + rel_freq_indiv[key_elem[0]] * rel_freq_indiv[key_elem[1]])
        return new_cooc

    # Prints current state of matrix.
    def _print_state(self):
        for key in self.cooc:
            key_parts = key.split(self.separator)
            print(f"{key_parts[0]:<5}|{key_parts[1]:<5}:{self.cooc[key]}")

    def save_matrix(self, filename):
        with open(filename, 'w+') as f:
            json.dump(self.cooc, f, sort_keys=True)


    def _load_matrix(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data


def main(args):
    if args.do_train:
        if not args.matrix_file:
            # Initialize the detector.
            detector = RhymeDetector(args.perfect_only)
        else:
            detector = RhymeDetector(args.perfect_only, args.matrix_file)
        detector.load_and_preprocess_data_from_file(args.train_file)
        detector.save_matrix('data/cooc_init.json', detector.cooc)
        i = 0
        changed = True
        # Train the detector until it stops changing or desired number of iterations is reached.
        while i < args.n and changed:
            print(f"ITERATION {i+1}")
            stats = detector.find_rhymes()
            changed = detector.adjust_matrix(stats)
            detector.save_matrix('data/cooc_iter'+str(i)+'.json', detector.cooc)
            i += 1
    if args.do_test:
        # Test the detector.
        if not args.do_train:
            matrix_file = args.matrix_file
            detector = RhymeDetector(args.perfect_only, matrix_file)
        test_data_pron = detector.load_and_preprocess_data_from_file(args.test_file)
        with open(args.test_file) as input:
            test_data = json.load(input)
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
    parser.add_argument('--verbose',  default=False, action='store_true')
    parser.add_argument('--n',  default=1000, help="Maximal number of iterations (actual number may be smaller if consistency is reached).")
    parser.add_argument('--train_file', required='do_train' in sys.argv, help="A file with songs for training.")
    parser.add_argument('--test_file', required='do_test' in sys.argv)
    parser.add_argument('--matrix_file', help="Matrix loaded for testing. If training selected, this matrix will be loaded as initialization matrix.")
    parser.add_argument('--perfect_only', default=False, action='store_true')
    # args = parser.parse_args([# '--train_file', 'data/train_lyrics0.001.json',
    #                           '--test_file', 'data/test_lyrics0.001.json',
    #                           '--matrix_C_file', 'data/matrixC_identity.csv',
    #                           '--matrix_V_file', 'data/matrixV_identity.csv',
    #                           # '--do_train',
    #                           '--do_test',
    #                           '--perfect_only'
    # ])
    args = parser.parse_args([
                                # '--train_file', 'data/train_lyrics0.001.json',
                                '--test_file', 'data/dev_lyrics0.001.json',
                                '--matrix_file', 'data/cooc_iter0.json',
                                # '--do_train',
                                # '--perfect_only',
                                '--do_test'
                                ])
    main(args)
