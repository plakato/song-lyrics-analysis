# -*- coding: utf-8 -*-
import ast
import json
import os
import re
import shlex
import subprocess
from os.path import isfile, join
from analyze_data import save_dataset
import xml.etree.ElementTree as ET


def create_sparsar_input_file_from_song(song, output_filename):
    with open(output_filename, 'w+') as output:
        output.write(song['title'] + '\n')
        output.write('by ' + song['artist'] + '.\n\n')
        for i in range(len(song['lyrics'])):
            line = song['lyrics'][i].strip()
            punctuation = '.'
            if line[-1] == '?' or \
                    line[-1] == '!' or \
                    line[-1] == '…' or \
                    line == '':
                punctuation = ''
            output.write(line + punctuation + '\n')


def replace_prohibited_characters(filename):
    with open(filename) as f:
        songs = json.load(f)
        for song in songs:
            for i in range(len(song['lyrics'])):
                song['lyrics'][i] = song['lyrics'][i]\
                    .replace('’', "'")\
                    .replace('"', '')
    save_dataset(songs, filename)


def run_sparsar_for_files(filename, prefix):
    with open(filename) as input:
        data = json.load(input)
        os.chdir("./sparsar_experiments")
    succeses = 0
    fails = 0
    for song in data:
        # SPARSAR only understands ENGLISH.
        if song['lang'] != 'ENGLISH':
            print('Song', song['title'], 'is not in English.')
            continue
        song_file = prefix + song['title'] + '.txt'
        print("Analyzing", song_file, '...')
        create_sparsar_input_file_from_song(song, song_file)
        # Don't get stuck on infinite sparsar executions.
        command_line = "./sparsar_man loadallouts -- \"" + song_file + "\""
        args = shlex.split(command_line)
        FNULL = open(os.devnull, 'w')
        try:
            completed = subprocess.run(args, stdout=FNULL, timeout=180)
            if completed.returncode == 0:
                succeses += 1
                print('Successfully analyzed', song_file)
            else:
                fails += 1
                print('Analysis of', song_file, 'failed.')
        except subprocess.TimeoutExpired:
            print('Analysis of', song_file, 'ran too long.')
            fails += 1

    print('Successfully analyzed', succeses, 'files.')
    print('Failed analyzing', fails, 'files.')


def get_scheme_letters(inputfile):
    tree = ET.parse(inputfile)

    root = tree.getroot()
    # Parse rhyme scheme.
    # Change Prolog format into JSON to be parsed as a dictionary.
    scheme = root[2][0].attrib['Stanza-based_Rhyme_Schemes']
    scheme = scheme.replace('-', '\":\"'
                                 '').replace('[', '{\"').replace(']',
                                                                 '\"}').replace(
        ',', '\",\"')
    scheme = '{\"scheme\":' + scheme.replace('\"{', '{').replace('}"',
                                                                 '}') + '}'
    # Get rid of empty values with regex.
    scheme = re.sub('\".?\":\{\"\"\},', '', scheme)
    scheme = json.loads(scheme)
    scheme_letters = {}
    for stanza in scheme['scheme'].values():
        scheme_letters.update(stanza)
    return scheme_letters, root


def extract_rhymes_to_csv(filename):
    inputfile = 'sparsar_experiments/outs/' + filename + '_phon.xml'
    outputfile = 'sparsar_experiments/rhymes/' + filename + '.csv'
    scheme_letters, root = get_scheme_letters(inputfile)
    print(scheme_letters)
    # Create output.
    keys = list(scheme_letters.keys())
    scheme_letter_no = -1
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
                scheme_letter_no += 1
                output.write('{0};{1};{2};{3}\n'.format(scheme_letters[keys[
                    scheme_letter_no]],
                                                        no,
                                                        ' '.join(words),
                                                        ' '.join(phons)))


def main():
    prefixes = ['', 'shuffled0_']
    for prefix in prefixes:
        filename = 'data/shuffled_lyrics/' + \
                   prefix + '100ENlyrics_cleaned.json'
        replace_prohibited_characters(filename)
        os.environ["PYTHONIOENCODING"] = "utf-8"
        run_sparsar_for_files(filename, prefix)
    path = 'sparsar_experiments/outs/'
    for item in os.listdir(path):
        if isfile(join(path, item)) and item.endswith('_phon.xml'):
            print('Extracting to csv...', item)
            extract_rhymes_to_csv(item[:-9])


if __name__ == '__main__':
    main()
