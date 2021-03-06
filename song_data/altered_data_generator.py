import copy
import os
import nltk

from analyze_data import save_dataset
from sparsar_analysis import get_scheme_letters, sparsar_process_song, get_scheme_letters,get_sparsar_phon_files_from_dir
import json
import csv
import random


# From random verses remove one or more words.
# prob - the probability that the verse will be changed
# to_remove - the fraction of words on line to be removed
def remove_words(path, filename, to_remove=0.3, prob=0.2):
    with open(path + filename) as original_file:
        songs = json.load(original_file)
    for song in songs:
        new_lyrics = []
        for line in song['lyrics']:
            if random.random() < prob:
                words = line.split(' ')
                n_words_to_remove = max(1, int(to_remove * len(words)))
                line = ' '.join(words[n_words_to_remove:])
            new_lyrics.append(line)
        song['lyrics'] = new_lyrics
    save_dataset(songs, path + 'remove_words' + '_' + filename)


# Switch each noun with preceding word.
def shift_nouns_back(path, filename):
    with open(path + filename) as original_file:
        songs = json.load(original_file)
    for song in songs:
        new_lyrics = []
        for line in song['lyrics']:
            tokens = nltk.word_tokenize(line)
            tags = nltk.pos_tag(tokens)
            new_line = []
            i = 0
            while i < len(tags)-1:
                if tags[i+1][1] == 'NN':
                    new_line.append(tags[i+1][0])
                    new_line.append(tags[i][0])
                    i += 1
                else:
                    new_line.append(tags[i][0])
                i += 1
            if i == len(tags) - 1:
                new_line.append(tags[-1][0])
            new_lyrics.append(' '.join(new_line))
        song['lyrics'] = new_lyrics
    save_dataset(songs, path + 'shift_noun_back' + '_' + filename)


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


def replace_random_words(path, filename, replace_word='giraffe', prob=0.33):
    with open(path + filename) as original_file:
        songs = json.load(original_file)
    modified_songs = songs.copy()
    for song in modified_songs:
        for i in range(len(song['lyrics'])):
            words = song['lyrics'][i].split(' ')
            for j in range(len(words)):
                if random.random() < prob:
                    words[j] = replace_word
            song['lyrics'][i] = ' '.join(words)
    save_dataset(modified_songs, path + 'replaced_word' + str(prob) + filename)


# Only first half of the file is shuffled.
def generate_halfmixed_lines(path, original_filename):
    with open(path + original_filename) as original_file:
        songs = json.load(original_file)
    shuffled_songs = copy.deepcopy(songs)
    for i in range(5):
        for song in shuffled_songs:
            count = int(len(song['lyrics'])/2)
            # Shuffled indexes for the first half of the lines.
            shuffled = shuffle(range(count))
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
path = 'test_sets/'
filename = '100ENlyrics_cleaned.json'
remove_words(path, filename)
# shift_nouns_back(path, filename)
# replace_random_words('data/shuffled_lyrics/', '100ENlyrics_cleaned.json')
# compare_line_shuffle()
# generate_mixed_lines('data/shuffled_lyrics/', '100ENlyrics_cleaned.json')
# generate_halfmixed_lines('data/shuffled_lyrics/', '100ENlyrics_cleaned.json')
# path = 'sparsar_experiments/outs/'
# for filename in os.listdir(path):
#     if filename.endswith('_phon.xml'):
#         score = calculate_rhyme_score(path + filename)
#         print(score, '\n')
