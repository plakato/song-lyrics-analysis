import json
import pycld2 as cld2


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
    with open(output_file, 'w+') as output:
        output.write('[\n')
        i = 0
        for song in unique:
            if i != 0:
                output.write(',\n')
            json.dump(song, output)
            i += 1
        output.write('\n]')


def filter_short(input_file):
    with open(input_file) as file:
        data = json.load(file)
        corrupted = 0
        total = 0
        for song in data:
            if len(song['lyrics']) < 10:
                # print(song['lyrics'])
                corrupted += 1
            total += 1
        print('corrupted: ', corrupted)
        print('total: ', total)


def detect_languages(input_file):
    with open(input_file, encoding="utf-8") as file:
        data = json.load(file)
        for song in data:
            try:
                lyrics = ''.join(song['lyrics'])
                isReliable, textBytesFound, details = cld2.detect(lyrics)
                print(details[0][0])
            except:
                with open('test.json', 'w+') as result:
                    json.dump(song, result)
                print('ERROR', song['lyrics'])
                print('ERROR', lyrics)


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
    # filter_short('lyrics_cleaned_unique.json')
    # detect_languages('lyrics_sample.json')
    print_statistics('lyrics_cleaned_unique.json')


if __name__== "__main__":
    main()


