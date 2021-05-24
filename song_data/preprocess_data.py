import os
import json
import re
import cld3


# Keep only is_music
# Add language.
# Add song ID.
# Keep only useful attributes.
# Remove prohibited characters.
# Remove shorter than 10 lines.
def isolate_relevant_songs_and_their_attributes(input_file, output_file, remove_file):
    with open(input_file) as input:
        data = json.load(input)
        new_data = []
        # Save removed data to check nothing useful was deleted.
        removed_data = []
        idx = 0
        deleted_is_music = 0
        lang_detection_errors = 0
        total_songs = 0
        not_reliable_eng = 0
        selected_attr = ['lyrics', 'title', 'album', 'genre', 'artist', 'url', 'year']
        for song in data:
            total_songs += 1
            # Remove what is not a music lyrics.
            if 'is_music' in song and (song['is_music'] != 'true') or song['lyrics'] == 'N/A':
                deleted_is_music += 1
                song['deletion_info'] = 'NOT MUSIC'
                removed_data.append(song)
                continue
            # Identify language - remove when language not identified.
            try:
                lyrics = ''.join(song['lyrics'])
                lang, prob, isReliable, proportion = cld3.get_language(lyrics)
                if lang == 'en' and not isReliable:
                    not_reliable_eng += 1
                prob = 0
            except:
                lang = 'DETECTION_ERROR'
                lang_detection_errors += 1
                song['deletion_info'] = 'LANGUAGE DETECTION ERROR'
                removed_data.append(song)
                continue
            new_song = {attr: song[attr] for attr in selected_attr}
            # Replace characters prohibited by libraries used. (SPARSAR, syllabify, CMUdict)
            for i in range(len(song['lyrics'])):
                new_song['lyrics'][i] = song['lyrics'][i].replace('’', "'").replace('"', '').strip()
            new_song['lang'] = lang
            new_song['id'] = idx
            new_song['title'] = song['title'].replace('/', '')
            new_data.append(new_song)
            idx += 1
    # Save new data.
    save_dataset(new_data, output_file)
    # Save removed data.
    save_dataset(removed_data, remove_file)
    # Print statistics.
    print(f"Analyzed {total_songs} songs.")
    print(f"Removed {len(removed_data)} songs.")
    print(f"Songs removed for is_music=False: {deleted_is_music}")
    print(f"Songs deleted for language detection errors: {lang_detection_errors}")
    print(f"English language detection not reliable for {not_reliable_eng} songs.")
    print(f"Saved {len(new_data)} valid songs.")


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



# Get rid of unnecessary words in lyrics.
def remove_unwanted_words_from_lyrics(lyrics):
    undesirable_words = ['verse', 'chorus', 'intro', 'outro', 'repeat', 'hook',
                         'bridge', 'transition', 'solo', 'http', 'www.'
                         'interlude']
    clean_lyrics = []
    newline = False
    for line in lyrics:
        # Get rid of lines with descriptive words (not part of lyrics).
        low_line = line.strip().lower()
        if any(re.search(x, low_line) for x in undesirable_words):
            words = low_line.split(' ')
            # Too short lines probably don't contain lyrics, just verse
            # description.
            if len(words) < 3:
                line = ''
            else:
                # There are lyrics following this word, just delete the
                # word, keep lyrics.
                words = line.split()
                if words[0] in undesirable_words:
                    # if the second word contains number, it's the number of the verse, remove it as well
                    if bool(re.search(r'\d', words[1])):
                        line = ' '.join(words[2:])
                    else:
                        line = ' '.join(words[1:])
                elif words[1] in undesirable_words:
                    line = ' '.join(words[2:])
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
        cleaned = remove_unwanted_words_from_lyrics(original)
        with open(miniset_cleaned_dir + file) as f:
            golden_cleaned = f.readlines()[2:]
        i = 0
        for i in range(len(golden_cleaned)):
            if golden_cleaned[i].strip() != cleaned[i].strip():
                print('CLEANED:', cleaned[i])
                print('CORRECT:', golden_cleaned[i])


# Extract into separate directory songs that do not appear to be valid songs.
def extract_suspicious_songs(input_file, susp_songs_file):
    with open(input_file) as input:
        data = json.load(input)
    suspicious = []
    fine = []
    for song in data:
        if song['word_count'] < 30:
            suspicious.append(song)
        else:
            fine.append(song)
    print('OK songs:', len(fine))
    print('Suspicious songs:', len(suspicious))
    save_dataset(fine, input_file)
    save_dataset(suspicious, susp_songs_file)


def filter_unique(input_file, output_file):
    with open(input_file) as file:
        data = json.load(file)
        total = 0
        unique_lyrics = set()
        unique = []
        duplicates = 0
        for song in data:
            lyrics = ''.join(song['lyrics'])
            if not lyrics in unique_lyrics:
                unique.append(song)
            else:
                duplicates += 1
            unique_lyrics.add(lyrics)
            total += 1
        print('total: ', total)

    print('Unique: ', len(unique))
    print(f'Removed {duplicates} duplicates.')
    save_dataset(unique, output_file)


def filter_english(input_file, output_file):
    with open(input_file) as file:
        data = json.load(file)
        total = 0
        english = []
        removed = 0
        for song in data:
            if song['lang'] == 'en':
                english.append(song)
            else:
                removed += 1
            total += 1
    print('Total: ', total)
    print('English: ', len(english))
    print(f'Removed {removed} non-English songs.')
    save_dataset(english, output_file)


# The constants are set from histograms of song length to filter out the outliers/extremes.
def filter_optimal_length(input_file, output_file):
    with open(input_file) as file:
        data = json.load(file)
        total = 0
        filtered = []
        removed_short = 0
        removed_long = 0
        for song in data:
            len_lines = len(song['lyrics'])
            len_chars = sum(len(i) for i in song['lyrics'])
            if len_chars < 65 or len_lines < 2.5 or song['word_count'] < 10:
                removed_short += 1
                continue
            if len_chars > 120000 or len_lines > 2000 or song['word_count'] > 21000:
                removed_long += 1
                continue
            filtered.append(song)
            total += 1
    print('Total: ', total)
    print('Filtered: ', len(filtered))
    print(f'Removed {removed_short} songs for being too short.')
    print(f'Removed {removed_long} songs for being too long.')
    save_dataset(filtered, output_file)


# Save in
# json format.
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
            song['word_count'] = total_words
    save_dataset(songs, filename)


def remove_unwanted_words_from_all_songs(file):
    with open(file) as f:
         songs = json.load(f)
    for song in songs:
        song['lyrics'] = remove_unwanted_words_from_lyrics(song['lyrics'])
    save_dataset(songs, file)

# Create separate file for different genres.
def split_by_genre(filename):
    genres = {}
    with open(filename) as input:
        data = json.load(input)
        for song in data:
            if song['genre'] not in genres:
                genres[song['genre']] = [song]
            else:
                genres[song['genre']].append(song)
    for genre in genres:
        result = re.search(r"(^.+)(\..+$)", filename)
        prefix, suffix = result.groups()
        genre_filename = prefix + '_' + genre + suffix
        save_dataset(genres[genre], genre_filename)


def create_individual_song_files(filename, output_dir, n=None):
    with open(filename) as input:
        data = json.load(input)
        i = 0
        for song in data:
            i += 1
            # Take just n songs.
            if n and i > n:
                continue
            # Get rid of slashes because they can affect the file location.
            song_file = output_dir + song['title'].replace('/', '') + '.txt'
            if not os.path.exists(song_file):
                print(f"Generating file for song {song_file}.")
                with open(song_file, 'w+') as output:
                    for line in song['lyrics']:
                        output.write(line + '\n')


def create_clean_dataset():
    isolate_relevant_songs_and_their_attributes('data/lyrics_original.json',
                                                'data/lyrics_cleaned.json',
                                                'data/removed_lyrics.json')
    filter_unique('data/lyrics_cleaned.json', 'data/lyrics_cleaned_unique.json')
    filter_english('data/lyrics_cleaned_unique.json', 'data/ENlyrics_cleaned_unique.json')
    remove_unwanted_words_from_all_songs('data/ENlyrics_cleaned_unique.json')
    add_word_count('data/ENlyrics_cleaned_unique.json')
    filter_optimal_length('data/ENlyrics_cleaned_unique.json', 'data/ENlyrics_final.json')
    split_by_genre('data/ENlyrics_final.json')


if __name__=='__main__':
    # create_clean_dataset()
    inputs = ['data/ENlyrics_final_pop.json',
              'data/ENlyrics_final_rock.json',
              'data/ENlyrics_final_rap.json',
              'data/ENlyrics_final_r-b.json',
              'data/ENlyrics_final_country.json']
    outputs = ['../evaluation/data/scheme_annotated/pop/',
               '../evaluation/data/scheme_annotated/rock/',
               '../evaluation/data/scheme_annotated/rap/',
               '../evaluation/data/scheme_annotated/r-b/',
               '../evaluation/data/scheme_annotated/country/']
    for i in range(len(inputs)):
        create_individual_song_files(inputs[i], outputs[i], 10)
# check_cleaner_with_miniset('data/miniset_for_lyrics_cleaning/',
#                            'data/miniset_for_lyrics_cleaning/manually_cleaned/')
# clean_all_songs('data/lyrics_cleaned.json')

# file = '100000ENlyrics_cleaned.json'
# extract_suspicious_songs('data/' + file, 'data/suspicious_' + file)


