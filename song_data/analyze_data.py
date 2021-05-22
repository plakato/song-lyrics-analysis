import csv
import json
import os
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


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
        plt.scatter(x, y, s=sizes, alpha=0.3, color='r', linewidths=0.0)
    # mpl.xlim(0, 70)
    # mpl.ylim(0,35)
    plt.xlabel('Verses')
    plt.ylabel('Rhyme classes')
    plt.title('Relationship between verse count and rhyme class count')
    plt.savefig('graphs/verse_count_vs_rhyme_class_count.png', dpi=300)


def create_histogram_for_length(input_file):
    songs_char_len = []
    songs_words = []
    songs_lines = []
    # Generate data.
    with open(input_file) as file:
        data = json.load(file)
    for song in data:
        one_string = ' '.join(song['lyrics'])
        songs_char_len.append(len(one_string))
        songs_words.append(len(one_string.split(' ')))
        songs_lines.append(len(song['lyrics']))
    # Draw graph for length in characters.
    bins = np.geomspace(1, max(songs_char_len), num=200)
    plt.hist(x=songs_char_len, bins=bins, color='#abd7eb')
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(x=20, color='red')
    plt.axvline(x=45000, color='red')
    plt.xlabel('Length in characters')
    plt.ylabel('No. of songs with given length')
    plt.title('Histogram of length in characters')
    plt.savefig('graphs/histogram_song_length_in_char.png')
    plt.show()
    # Draw graph for length in characters.
    bins = np.geomspace(1, max(songs_words), num=200)
    plt.hist(x=songs_words, bins=bins, color='#abd7eb')  # 0504aa
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(x=10, color='red')
    plt.axvline(x=9000, color='red')
    plt.xlabel('Length in words')
    plt.ylabel('No. of songs with given length')
    plt.title('Histogram of length in words')
    plt.savefig('graphs/histogram_song_length_in_words.png')
    plt.show()
    # Draw graph for length in lines.
    bins = np.geomspace(1, max(songs_lines), num=200)
    plt.hist(x=songs_lines, bins=bins, color='#abd7eb')  # 0504aa
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(x=6, color='red')
    plt.axvline(x=2000, color='red')
    plt.xlabel('Length in lines')
    plt.ylabel('No. of songs with given length')
    plt.title('Histogram of length in lines')
    plt.savefig('graphs/histogram_song_length_in_lines.png')
    plt.show()


def print_short_and_long(input_file, short=10, long=30000):
    with open(input_file) as file:
        data = json.load(file)
        short_count = 0
        long_count = 0
        total = 0
        for song in data:
            chars = len(''.join(song['lyrics']))
            if len(song['lyrics']) < short:
                print("SHORT:", song['lyrics'])
                short_count += 1
            elif chars > long:
                print("LONG:", song['title'], "CHARS:", chars)
                long_count += 1
            total += 1
        print('Shorter than {0} lines: {1}'.format(short, short_count))
        print('Longer than {0} characters: {1}'.format(long, long_count))
        print('Total: ', total)


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
                        song[attr] == 'N/A' or \
                        song[attr] == 'null'):
                    attributes_count[attr] += 1
            lang = song['lang']
            if lang in languages:
                languages[lang] += 1
            else:
                languages[lang] = 1
    # Sort language by count.
    languages = dict(sorted(languages.items(), key=lambda item: item[1]))
    save_piechart_with_genres(genres)
    print('Total songs: ', total_songs)
    print('Average number of lines per song: ', total_lines/total_songs)
    print('Average number of words per line: ', total_words/total_lines)
    print('Genres: \n', genres)
    print('Attributes (count of non-empty values):\n', attributes_count)
    print('Languages:\n', len(languages))
    print(languages)
    print('Undetected: ', undetected_lang)


def save_piechart_with_genres(genres):
    # Data to plot
    labels = genres.keys()
    sizes = genres.values()
    colors = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    # Plot
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.legend(wedges, labels, title="Genres", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8)
    plt.axis('equal')
    plt.savefig('graphs/piechart_genres.png')
    plt.show()


def main():
    # create_histogram_for_length('data/ENlyrics_cleaned_unique.json')
    print_statistics('data/ENlyrics_final_country.json')


if __name__== "__main__":
    main()


