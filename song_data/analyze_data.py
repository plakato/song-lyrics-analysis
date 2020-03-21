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
        for song in unique:
            json.dump(song, output)
            output.write(',\n')
        output.write(']')


def filter_short(input_file, output_file):
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
        for song in data:
            isReliable, textBytesFound, details = cld2.detect(''.join(song['lyrics']))
            print(details[0])


def main():
    # filter_unique('lyrics_sample.json', 'lyrics_cleaned_unique.json')
    detect_languages('lyrics_sample.json')


if __name__== "__main__":
    main()


