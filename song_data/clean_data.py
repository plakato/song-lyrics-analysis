import os

from sparsar_analysis import create_sparsar_input_file_from_song
from analyze_data import save_dataset
import json
import re
import numpy as np


# Keep only is_music
# Add "lang", "very_short"
# Keep only useful attributes
def create_clean_dataset(input_file, output_file):
    with open(input_file) as input:
        data = json.load(input)
        new_data = []
        selected_attr = ['lyrics', 'title', 'album', 'genre', 'artist', 'url', 'rg_artist_id', 'rg_type', 'rg_tag_id', 'rg_song_id', 'rg_album_id', 'rg_created', 'has_featured_video', 'has_verified_callout', 'has_featured_annotation']
        for song in data:
            if song['is_music'] != 'true':
                continue
            try:
                lyrics = ''.join(song['lyrics'])
                isReliable, textBytesFound, details = cld2.detect(lyrics)
                lang = details[0][0]
            except:
                lang = 'DETECTION_ERROR'
            very_short = False
            if len(song['lyrics']) < 10:
                very_short = True
            new_song = { attr: song[attr] for attr in selected_attr }
            new_song['very_short'] = very_short
            new_song['lang'] = lang
            new_data.append(new_song)
    # Print new data.
    with open(output_file, 'w+') as output:
        output.write('[\n')
        i = 0
        for song in new_data:
            if i != 0:
                output.write(',\n')
            json.dump(song, output)
            i += 1
        output.write('\n]')


def detect_noise(input_file):
    with open(input_file) as input:
        data = json.load(input)
        noise = set()
        for song in data:
            for line in song['lyrics']:
                parsed = line.split(' ')
                if len(parsed) < 3:
                    noise.add(line)
                # re.match('\*\**\*\*')

    print(noise)


# Input: file with multiple songs in json
# Creates a separate file for each song.
def create_individual_files(filename):
    with open(filename) as input:
        data = json.load(input)
        i = 0
        for song in data:
            if song['lang'] != 'ENGLISH':
                continue
            i += 1
            # Take just hundred songs.
            if i < 200 or i > 300:
                continue
            # Get rid of slashes because they can affect the file location.
            song_file = song['title'].replace('/', '') + '.txt'
            print(song_file)
            create_sparsar_input_file_from_song(song,
                                                'data/individual_songs/' +
                                                song_file)


# Get rid of unnecessary words in lyrics.
def clean_song(lyrics):
    undesirable_words = ['verse', 'chorus', 'intro', 'outro', 'repeat', 'hook',
                         'bridge', 'transition', 'solo', 'http', 'www.']
    clean_lyrics = []
    newline = False
    for line in lyrics:
        # Get rid of lines with descriptive words (not part of lyrics).
        low_line = line.strip().lower()
        if any(re.search(x, low_line) for x in undesirable_words):
            words = low_line.split(' ')
            if len(words) < 5:
                line = ''
            else:
                # There are lyrics following this word, just delete the
                # word, keep lyrics.
                line = ''.join(line.split(':')[1:])
        # Get rid of multiple newlines.
        if line.strip() == '':
            if newline:
                continue
            else:
                newline = True
        else:
            newline = False
        clean_lyrics.append(line)
    return clean_lyrics


# Use mini dataset of common mistakes to check if the cleaner works properly.
def check_cleaner_with_miniset(miniset_dir, miniset_cleaned_dir):
    files = [f for f in os.listdir(miniset_dir)
             if os.path.isfile(os.path.join(miniset_dir, f))]
    for file in files:
        print(file)
        with open(miniset_dir + file) as f:
            original = f.readlines()[2:]
        cleaned = clean_song(original)
        with open(miniset_cleaned_dir + file) as f:
            golden_cleaned = f.readlines()[2:]
        i = 0
        for i in range(len(golden_cleaned)):
            if golden_cleaned[i].strip() != cleaned[i].strip():
                print('CLEANED:', cleaned[i])
                print('CORRECT:', golden_cleaned[i])


# Extract into separate directory songs that do not appear to be valid songs.
def extract_suspicious_songs(input_file):
    with open(input_file) as input:
        data = json.load(input)
        artist_counts = {}
        for song in data:
            if song['artist'] not in artist_counts:
                artist_counts[song['artist']] = [song['words']]
            artist_counts[song['artist']].append(song['words'])
            if song['words'] < 50:
                print(song['title'])
        print(artist_counts)
        k = 2
        for artist in artist_counts:
            values = artist_counts[artist]
            mean = np.mean(values)
            s2 = sum(np.square(values))
            sigma = np.sqrt(s2/len(values) - mean**2)
            left_margin = mean - k*sigma
            right_margin = mean + k*sigma
            for value in values:
                if value < left_margin: # or value > right_margin:
                    print(artist, ': ', value)


def clean_all_songs(file):
    with open(file) as f:
        songs = json.load(f)
    for song in songs:
        song['lyrics'] = clean_song(song['lyrics'])
    save_dataset(songs, file)

# # Create individual song files for easier song checking by eye.
# filename = 'data/1000ENlyrics_cleaned.json'
# create_individual_files(filename)

# check_cleaner_with_miniset('data/miniset_for_lyrics_cleaning/',
#                            'data/miniset_for_lyrics_cleaning/manually_cleaned/')

# extract_suspicious_songs('data/1000ENlyrics_cleaned.json')


clean_all_songs('data/lyrics_cleaned.json')

