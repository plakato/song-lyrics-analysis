import copy
import os

from analyze_data import save_dataset
from sparsar_analysis import get_scheme_letters, sparsar_process_song, get_scheme_letters,get_sparsar_phon_files_from_dir
import json
import csv
import random


def shuffle(data):
    return random.sample(data, len(data))


def generate_mixed_lines(path, original_filename, n=1):
    with open(path + original_filename) as original_file:
        songs = json.load(original_file)
    shuffled_songs = songs.copy()
    for i in range(n):
        for song in shuffled_songs:
            shuffled = shuffle(song['lyrics'])
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


# An attempt to create a score of how much rhyme-y the song is.
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


# An experiment to compare SPARSAR performance - difference between shuffling data BEFORE and AFTER sparsar analysis.
# Expects to have unshuffled data analyzed already.
def compare_line_shuffle():
    random.seed(12345)
    # Generate shuffled data.
    path = 'sparsar_experiments/line_shuffle_comparison/'
    original_filename = 'songs_for_comparison.json'
    with open(path + original_filename) as original_file:
        songs = json.load(original_file)
    shuffled_songs = songs.copy()
    os.chdir("./sparsar_experiments/line_shuffle_comparison/analyzed_after_shuffle")
    for song in shuffled_songs:
        shuffled = shuffle(song['lyrics'])
        print(shuffled)
        song['lyrics'] = shuffled
        # Run sparsar for shuffled.
        result = sparsar_process_song(song, prefix='shuffled0')
        if result == 1:
            print('Failed to analyze', song['title'])
    # Get shuffled schemes.
    shuffled_schemes = {}
    files, names = get_sparsar_phon_files_from_dir('outs')
    names = [name.replace('shuffled0', '') for name in names]
    for i in range(len(files)):
        scheme, _ = get_scheme_letters('outs/' + files[i])
        shuffled_schemes[names[i]] = list(scheme.values())
    # Get original schemes and shuffle them.
    # Re-initialize seed to get the same shuffle.
    random.seed(12345)
    os.chdir("../analyzed_before_shuffle")
    original_schemes = {}
    for i in range(len(files)):
        scheme, _ = get_scheme_letters(files[i].replace('shuffled0', ''))
        original_schemes[names[i]] = shuffle(list(scheme.values()))
    print('SHUFFLED AFTER\n', original_schemes)
    print('SHUFFLED BEFORE\n', shuffled_schemes)


random.seed(12345)
# compare_line_shuffle()
# generate_mixed_lines('data/shuffled_lyrics/', '100ENlyrics_cleaned.json')
# generate_halfmixed_lines('data/shuffled_lyrics/', '100ENlyrics_cleaned.json')
# path = 'sparsar_experiments/outs/'
# for filename in os.listdir(path):
#     if filename.endswith('_phon.xml'):
#         score = calculate_rhyme_score(path + filename)
#         print(score, '\n')
