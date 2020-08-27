import csv
import json
import os
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as mpl
from collections import Counter
import pandas as pd


# Test if the data can be modeled using Chinese restaurant process.
def test_CRP(dir, duplicates=False):
    n_CRP_won = 0
    n_CRP_lost = 0
    too_long = 0
    for item in os.listdir(dir):
        file_name = join(dir, item)
        if isfile(file_name) and item.endswith('.csv'):
            print(file_name)
            df = pd.read_csv(file_name, sep=';')
            letters = df['Rhyme Scheme Letter']
            if not duplicates:
                lyrics = df['Lyrics']
                unique = set()
                for i in range(len(lyrics)-1, -1, -1):
                    if lyrics[i] in unique:
                        del letters[i]
                    unique.add(lyrics[i])
            CRP_prob, clusters = CRP_calculate_prob_of_outcome(letters)
            rand_prob = random_calculate_prob_of_outcome(letters)
            print('Verses: {0}, Classes: {1}, CRP-RND ratio: {2}\nCRP Probability: {3}\nRND Probability: {4}'.format(
                len(letters), len(clusters.keys()), CRP_prob/rand_prob if rand_prob > 0 else 'None', CRP_prob, rand_prob))
            if rand_prob == 0:
                too_long += 1
            elif CRP_prob/rand_prob > 1:
                n_CRP_won += 1
            else:
                n_CRP_lost += 1
    print('CRP won in {0} cases, lost in {1} cases. Undetectable (too long) in {2} cases'.format(n_CRP_won, n_CRP_lost, too_long))


def CRP_calculate_prob_of_outcome(seq, alpha=0.5):
    prob = 1.0
    clusters = {}
    for item in seq:
        total = sum(clusters.values()) + alpha
        if item in clusters:
            prob *= clusters[item]/total
            clusters[item] += 1
        else:
            prob *= alpha/total
            clusters[item] = 1
    return prob, clusters


# Calculate probability that the given sequence would appear randomly
def random_calculate_prob_of_outcome(seq):
    n_clusters = len(set(seq))
    prob = (1/n_clusters)**len(seq)
    return prob


def graph_verse_count_vs_rhyme_class_count(dir):
    # Count occurences of classes for verse counts.
    counts = {}
    for item in os.listdir(dir):
        file_name = join(dir, item)
        if isfile(file_name) and item.endswith('.csv'):
            with open(file_name, newline='') as song:
                rows = csv.reader(song, delimiter=';')
                next(rows)
                verses_count = 0
                classes = set()
                for row in rows:
                    verses_count += 1
                    classes.add(row[0])
                classes_count = len(classes)
                if verses_count not in counts:
                    counts[verses_count] = []
                counts[verses_count].append(classes_count)
    # Graph the result.
    for key in counts.keys():
        c = Counter(counts[key])
        # create a list of the sizes, here multiplied by 10 for scale
        sizes = [5 * c[key] for key in c]
        x = [key]*len(c)
        y = c.keys()
        mpl.scatter(x, y, s=sizes, alpha=0.3, color='r', linewidths=0.0)
    # mpl.xlim(0, 70)
    # mpl.ylim(0,35)
    mpl.xlabel('Verses')
    mpl.ylabel('Rhyme classes')
    mpl.title('Relationship between verse count and rhyme class count')
    mpl.savefig('graphs/verse_count_vs_rhyme_class_count.png', dpi=300)


def filter_unique(input_file, output_file):
    with open(input_file) as file:
        data = json.load(file)
        total = 0
        unique_lyrics = set()
        unique = []
        for song in data:
            lyrics = ''.join(song['lyrics'])
            if not lyrics in unique_lyrics:
                unique.append(song)
            unique_lyrics.add(lyrics)
            total += 1
        print('total: ', total)

    print('Unique: ', len(unique))
    save_dataset(unique, output_file)


# Save in json format.
def save_dataset(dataset, output_file):
    with open(output_file, 'w+') as output:
        output.write('[\n')
        i = 0
        for song in dataset:
            if i != 0:
                output.write(',\n')
            json.dump(song, output)
            i += 1
        output.write('\n]')


def add_word_count(filename):
    with open(filename) as f:
        songs = json.load(f)
        for song in songs:
            total_words = 0
            for line in song['lyrics']:
                words = line.strip().split()
                total_words += len(words)
            song['words'] = total_words
    save_dataset(songs, filename)


def filter_short(input_file):
    with open(input_file) as file:
        data = json.load(file)
        corrupted = 0
        total = 0
        for song in data:
            if len(song['lyrics']) < 10:
                print(song['lyrics'])
                corrupted += 1
            total += 1
        print('corrupted: ', corrupted)
        print('total: ', total)


def detect_languages(input_file):
    with open(input_file) as file:
        data = json.load(file)
        for i in range(len(data)):
            song = data[i]
            try:
                lyrics = ''.join(song['lyrics'])
                isReliable, textBytesFound, details = cld2.detect(lyrics)
                print(isReliable, details)
            except:
                # with open('test.json', 'w+') as result:
                #     json.dump(song, result)
                print('ERROR')# , song['lyrics'])
                # print('ERROR', lyrics)


def print_statistics(input_file):
    with open(input_file) as input:
        data = json.load(input)
        total_songs = 0
        total_lines = 0
        total_words = 0
        genres = {}
        languages = {}
        undetected_lang = 0
        for song in data:
            if total_songs == 0:
                attributes = song.keys()
                attributes_count = dict.fromkeys(attributes, 0)
            if song['is_music'] != 'true':
                continue
            total_songs += 1
            total_lines += len(song['lyrics'])
            for line in song['lyrics']:
                total_words += len(line.split(' '))
            if song['genre'] not in genres:
                genres[song['genre']] = 1
            else:
                genres[song['genre']] += 1
            for attr in attributes:
                if not (song[attr] == '' or \
                        song[attr] == [] or \
                        song[attr] == 'N/A'):
                    attributes_count[attr] += 1
            try:
                lyrics = ''.join(song['lyrics'])
                isReliable, textBytesFound, details = cld2.detect(lyrics)
                lang = details[0][0]
                if lang in languages:
                    languages[lang] += 1
                else:
                    languages[lang] = 1
            except:
                undetected_lang += 1

    print('Total songs: ', total_songs)
    print('Average number of lines per song: ', total_lines/total_songs)
    print('Average number of words per line: ', total_words/total_lines)
    print('Genres: \n', genres)
    print('Attributes:\n', attributes_count)
    print('Languages:\n', len(languages))
    print(languages)
    print('Undetected: ', undetected_lang)


def main():
    # filter_unique('lyrics_cleaned.json', 'lyrics_cleaned_unique.json')
    # filter_short('data/1000lyrics_cleaned.json')
    # detect_languages('data/100lyrics_cleaned.json')
    # print_statistics('lyrics_cleaned_unique.json')
    # create_clean_dataset('data/lyrics_cleaned_unique.json', 'data/
    # lyrics_cleaned2.json')
    # remove_noise('data/10000lyrics_cleaned.json')
    # add_word_count('data/lyrics_cleaned.json')
    # graph_verse_count_vs_rhyme_class_count('sparsar_experiments/rhymes/original')
    test_CRP('sparsar_experiments/rhymes/original')


if __name__== "__main__":
    main()


