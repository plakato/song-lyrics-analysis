import copy
import os

from analyze_data import save_dataset
from sparsar_analysis import get_scheme_letters
import json
import csv
import random


def generate_mixed_lines(path, original_filename):
    with open(path + original_filename) as original_file:
        songs = json.load(original_file)
    shuffled_songs = songs.copy()
    for i in range(5):
        for song in shuffled_songs:
            shuffled = random.sample(song['lyrics'], len(song['lyrics']))
            song['lyrics'] = shuffled
        save_dataset(shuffled_songs, path + 'shuffled' + str(i) + '_' +
                     original_filename)


def generate_halfmixed_lines(path, original_filename):
    with open(path + original_filename) as original_file:
        songs = json.load(original_file)
    shuffled_songs = copy.deepcopy(songs)
    for i in range(5):
        for song in shuffled_songs:
            count = int(len(song['lyrics'])/2)
            # Randomly pick half of the lines to be shuffled between
            # themselves.
            shuffled = random.sample(range(count), count)
            lines = song['lyrics'].copy()
            for j in range(count):
                song['lyrics'][j] = lines[shuffled[j]]
        save_dataset(shuffled_songs, path + 'halfshuffled' + str(i) + '_' +
                     original_filename)


def calculate_rhyme_score(filename):
    print(filename)
    scheme_letters = get_scheme_letters(filename)
    letter_locs = {}
    for loc in scheme_letters:
        if scheme_letters[loc] not in letter_locs:
            letter_locs[scheme_letters[loc]] = []
        letter_locs[scheme_letters[loc]].append(int(loc))
    difference = 0
    # Total number of pairs of lines with the same letter.
    pairs = 0
    for letter in letter_locs:
        locs = letter_locs[letter]
        pairs += len(locs) - 1
        # Sum distances between lines with the same letter.
        difference += locs[-1] - locs[0] - (len(locs) - 1)
    return difference/pairs if pairs != 0 else 0


# generate_halfmixed_lines('data/shuffled_lyrics/', '100ENlyrics_cleaned.json')
path = 'sparsar_experiments/outs/'
for filename in os.listdir(path):
    if filename.endswith('_phon.xml'):
        score = calculate_rhyme_score(path + filename)
        print(score, '\n')
