# -*- coding: utf-8 -*-
import ast
import json
import os
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
            output.write(line + '\n')


def run_sparsar_for_files(filename):
    with open(filename) as input:
        data = json.load(input)
        os.chdir("./sparsar_experiments")
        for song in data:
            if song['lang'] == 'ENGLISH':
                song_file = song['title'] + '.txt'
                print(song_file)
                create_sparsar_input_file_from_song(song, song_file)
                os.system("./sparsar_man loadallouts -- \"" + song_file + "\"")


def extract_rhymes_to_csv(filename):
    inputfile = 'sparsar_experiments/outs/\'' + filename + '\'_phon.xml'
    outputfile = 'sparsar_experiments/rhymes/\'' + filename + '\'.csv'
    tree = ET.parse(inputfile)

    root = tree.getroot()
    # Get rhyme scheme.
    schemes = root[2][0].attrib['Different_Rhyme_Schemes']
    schemes = schemes.replace('[', '').replace(']', '').split(',')
    print(schemes)
    # Remove title and author rhyme letters.
    i = 0
    with open(outputfile, 'w+') as output:
        output.write('Rhyme Scheme Letter;Line Number;Lyrics;Phonetic '
                     'Transcription\n')
        for stanza in root[0]:
            for line in stanza[1]:
                words = []
                phons = []
                for word in line.attrib['line_syllables'].split(']'):
                    if word == '':
                        continue
                    parts = word.replace('[', '').split('/')
                    words.append(parts[0].replace(',', ''))
                    phons.append(parts[1].replace(',', '_'))
                output.write('{0};{1};{2};{3}\n'.format(schemes[i],
                                                        line.attrib[
                    'no'],
                                                       ' '.join(
                    words), ' '.join(phons)))
                i += 1
        while i < len(schemes):
            output.write(schemes[i] + '\n')
            i += 1


def main():
    filename = 'data/100lyrics_cleaned.json'
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # run_sparsar_for_files(filename)
    extract_rhymes_to_csv('Cruising USA Lyrics.txt')


if __name__ == '__main__':
    main()
