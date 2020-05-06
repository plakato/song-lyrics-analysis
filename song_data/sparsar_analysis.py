# -*- coding: utf-8 -*-
import ast
import json
import os
from os.path import isfile, join
from analyze_data import save_dataset
import xml.etree.ElementTree as ET


def create_sparsar_input_file_from_song(song, output_filename):
    with open(output_filename, 'w+') as output:
        output.write(song['title'] + '\n')
        output.write('by ' + song['artist'] + '.\n\n')
        for i in range(len(song['lyrics'])):
            line = song['lyrics'][i] # .replace('-', '-')
            # line = line.replace(u'–', '-')
            # line = line.replace('\'', '\'')
            # line = line.replace('\'', '\'')
            output.write(line + '.\n')


def replace_prohibited_characters(filename):
    with open(filename) as f:
        songs = json.load(f)
        for song in songs:
            for i in range(len(song['lyrics'])):
                song['lyrics'][i] = song['lyrics'][i]\
                    .replace('’', "'")\
                    .replace('"', '')
    save_dataset(songs, filename)


def run_sparsar_for_files(filename):
    with open(filename) as input:
        data = json.load(input)
        os.chdir("./sparsar_experiments")
        for song in data:
            if song['lang'] == 'ENGLISH':
                song_file = 'shuffled' + song['title'] + '.txt'
                print(song_file)
                create_sparsar_input_file_from_song(song, song_file)
                os.system("./sparsar_man loadallouts -- \"" + song_file + "\"")


def get_scheme_letters(inputfile):
    tree = ET.parse(inputfile)

    root = tree.getroot()
    # Parse rhyme scheme.
    scheme = root[2][0].attrib['Stanza-based_Rhyme_Schemes']
    scheme = scheme.replace('-', '\":\"'
                                 '').replace('[', '{\"').replace(']',
                                                                 '\"}').replace(
        ',', '\",\"')
    scheme = '{\"scheme\":' + scheme.replace('\"{', '{').replace('}"',
                                                                 '}') + '}'
    scheme = json.loads(scheme)
    scheme_letters = {}
    for stanza in scheme['scheme'].values():
        scheme_letters.update(stanza)
    return scheme_letters


def extract_rhymes_to_csv(filename):
    inputfile = 'sparsar_experiments/outs/' + filename + '_phon.xml'
    outputfile = 'sparsar_experiments/rhymes/' + filename + '.csv'
    scheme_letters = get_scheme_letters(inputfile)
    print(scheme_letters)
    # Create output.
    with open(outputfile, 'w+') as output:
        output.write('Rhyme Scheme Letter;Line Number;Lyrics;Phonetic '
                     'Transcription\n')
        for stanza in root[0]:
            for line in stanza[1]:
                words = []
                phons = []
                no = int(line.attrib['no'])
                for word in line.attrib['line_syllables'].split(']'):
                    if word == '':
                        continue
                    parts = word.replace('[', '').split('/')
                    words.append(parts[0].replace(',', ''))
                    phons.append(parts[1].replace(',', '_'))
                # Adding 3 to account for first three lines (author, title,
                # empty line).
                output.write('{0};{1};{2};{3}\n'.format(scheme_letters[str(no +
                                                                       3)],
                                                        no,
                                                        ' '.join(words),
                                                        ' '.join(phons)))


def main():
    # filename = 'data/shuffled_lyrics/shuffled0_100ENlyrics_cleaned.json'
    # replace_prohibited_characters(filename)
    # os.environ["PYTHONIOENCODING"] = "utf-8"
    # run_sparsar_for_files(filename)
    path = 'sparsar_experiments/outs/'
    for item in os.listdir(path):
        if isfile(join(path, item)) and item.endswith('_phon.xml') \
                and item.startswith('\'shuffled'):
            print('Extracting to csv...', item)
            extract_rhymes_to_csv(item[:-9])


if __name__ == '__main__':
    main()
