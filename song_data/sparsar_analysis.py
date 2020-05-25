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
            # if line == '':
            #     output.write('.\n')
            # else:
            #     output.write(line + '\n')
            if line == '' or \
                    line[-1] == '?' or \
                    line[-1] == '!' or \
                    line[-1] == '…':
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


def run_sparsar_for_files(filename, prefix, start_idx=0,
                          failed_file='sparsar_failed_song_idxs.txt'):
    failed_idxs = []
    try:
        with open(filename) as input:
            data = json.load(input)
            os.chdir("./sparsar_experiments")
        succeses = 0
        fails = 0
        for i in range(len(data)):
            # Start analysis at given position.
            if i < start_idx:
                continue
            song = data[i]
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
                    failed_idxs.append(i)
                    print('Analysis of', song_file, 'failed.')
            except subprocess.TimeoutExpired:
                print('Analysis of', song_file, 'ran too long.')
                fails += 1
                failed_idxs.append(i)
            print("Finished analysis at index {0}. success {1} / fail {2} / total "
                  "{3}".format(i, succeses, fails, i-start_idx+1))

        print('Successfully analyzed', succeses, 'files.')
        print('Failed analyzing', fails, 'files.')
    except:
        with open(failed_file, 'a+') as error_output:
            for id in failed_idxs:
                error_output.write(str(id) + '\n')


# Change letters used in the rhyme scheme so that they are used alphabetically.
def normalize_rhyme_scheme(old_scheme):
    # Get next string alphabetically.
    def string_next(a):
        def next_char(char):
            return chr(ord(char) + 1)

        if a == '':
            return 'a'

        if a[-1] == 'z':
            return string_next(a[:-1]) + 'a'
        else:
            return a[:-1] + next_char(a[-1])

    current_str = ""
    rename_dict = {}
    new_scheme = []

    for l in old_scheme:
        if rename_dict.get(l) is None:
            new_letter = string_next(current_str)
            current_str = new_letter
            rename_dict[l] = new_letter
        new_scheme.append(rename_dict.get(l))
    return new_scheme


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
    prefix = ''
    if filename.startswith('shuffled'):
        prefix = filename[:9] + '/'
    else:
        prefix = 'original/'
    outputfile = 'sparsar_experiments/rhymes/' + prefix + filename + '.csv'
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
    # Prepare files for SPARSAR.
    prefixes = ['']
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for prefix in prefixes:
        filename = 'data/' + \
                   prefix + 'ENlyrics_cleaned2.json'
        # replace_prohibited_characters(filename)
        # Generate SPARSAR output files.
        run_sparsar_for_files(filename, prefix, 6422)
    # Extract useful information from SPARSAR output files to .csv file.
    # path = 'sparsar_experiments/outs/'
    # total = 0
    # for item in os.listdir(path):
    #     if isfile(join(path, item)) and item.endswith('_phon.xml'):
    #         print('Extracting to csv...', item)
    #         extract_rhymes_to_csv(item[:-9])
    #         total += 1
    # print('Generated', total, '.csv files.')


if __name__ == '__main__':
    main()
