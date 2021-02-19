# -*- coding: utf-8 -*-
from datetime import datetime
import json
import os
import re
import shlex
import subprocess
from os.path import join, isfile
from analyze_data import save_dataset
import xml.etree.ElementTree as ET
# import altered_data_generator

# VARIABLES
# File where indexes of songs for which SPARSAR analysis failed are written.
failed='sparsar_failed_song_idxs.txt'


def create_sparsar_input_file_from_song(song, output_filename):
    with open(output_filename, 'w+') as output:
        output.write(song['title'] + '\n')
        output.write('by ' + song['artist'] + '.\n\n')
        for i in range(len(song['lyrics'])):
            # Get rid of semicolons and & because the punctuator deletes
            # everything following a semicolon.
            song['lyrics'][i] = song['lyrics'][i].strip().replace(
                '&amp;', 'and').replace(';', ',')
            # punctuation = '.'
            # if line == '' or \
            #         line[-1] == '?' or \
            #         line[-1] == '!' or \
            #         line[-1] == '…':
            #     punctuation = ''
        punctuated_lyrics = add_punctuation(song['lyrics'])
        for i in range(len(punctuated_lyrics)):
            output.write(punctuated_lyrics[i] + '\n')


def replace_prohibited_characters(filename):
    with open(filename) as f:
        songs = json.load(f)
        for song in songs:
            for i in range(len(song['lyrics'])):
                song['lyrics'][i] = song['lyrics'][i]\
                    .replace('’', "'")\
                    .replace('"', '')
    save_dataset(songs, filename)


# Returns "success" = 0, "fail" = 1
def sparsar_process_song(song, prefix=''):
    # SPARSAR only understands ENGLISH.
    if song['lang'] != 'ENGLISH':
        print('Song', song['title'], 'is not in English. Skipping.')
        return 1
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
            print('Successfully analyzed', song_file)
            return 0
        else:
            print('Analysis of', song_file, 'failed.')
            return 1
    except subprocess.TimeoutExpired:
        print('Analysis of', song_file, 'ran too long.')
        return 1


def run_sparsar_for_failed(prefix, filename):
    failed_idxs = []
    total = 0
    with open(filename) as input:
        data = json.load(input)
    os.chdir("./sparsar_experiments")
    with open(failed) as indexes_file:
        indexes = indexes_file.readlines()
    indexes = list(filter(lambda a: a != '\n', indexes))
    unfixed = [int(i.strip()) for i in indexes]
    succeses = 0
    fails = 0
    try:
        for i in indexes:
            if i == '\n':
                continue
            i = int(i.strip())
            total += 1
            song = data[i]
            result = sparsar_process_song(song, prefix)
            if result == 0:
                succeses += 1
                unfixed.remove(i)
            else:
                fails += 1
                failed_idxs.append(i)
            print("Finished analysis at index {0}. success {1} / fail {2} / "
                "total {3}".format(i, succeses, fails, total))
    finally:
        with open(failed, 'w+') as error_output:
            for id in sorted(set(failed_idxs + unfixed)):
                if id is str and id != '\n':
                    id = id.strip()
                error_output.write(str(id).strip() + '\n')
    print('Successfully analyzed', succeses, 'files.')
    print('Failed analyzing', fails, 'files.')


def run_sparsar_for_files(filename, prefix, start_idx=0, end_idx=1000*1000,
                          failed_file=failed):
    failed_idxs = []
    with open(filename) as input:
        data = json.load(input)
        os.chdir("./sparsar_experiments")
    succeses = 0
    fails = 0
    try:
        for i in range(len(data)):
            # Start analysis at given position.
            if i < start_idx or i > end_idx:
                continue
            song = data[i]
            result = sparsar_process_song(song, prefix)
            if result == 0:
                succeses += 1
            else:
                fails += 1
                failed_idxs.append(i)
            print("Finished analysis at index {0}. success {1} / fail {2} / "
                "total {3}".format(i, succeses, fails, i - start_idx + 1))
    except:
        with open(failed_file, 'w+') as error_output:
            for id in failed_idxs:
                error_output.write(str(id) + '\n')
    print('Successfully analyzed', succeses, 'files.')
    print('Failed analyzing', fails, 'files.')


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
    # Mark '&' correctly because it is sometimes used incorrectly in generated
    # xmls.
    fixed_input = []
    with open(inputfile) as input:
        for line in input:
            fixed_input.append(re.sub('&(?!amp;)',  '&amp;', line))
    try:
        root = ET.fromstringlist(fixed_input)
    except Exception as error:
        print('Failed analyzing', inputfile)
        print(error)
        return None, None

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


def get_sparsar_phon_files_from_dir(dir):
    files = []
    names = []
    for filename in os.listdir(dir):
        if filename.endswith('_phon.xml'):
            name = filename.replace('\'', '').replace('.txt_phon.xml', '')
            files.append(filename)
            names.append(name)
    return files, names


def extract_rhymes_to_csv(path, filename, output_path):
    inputfile = path + filename
    filename = filename.replace('\'', '').replace(".txt_phon.xml", "")
    if filename.startswith('shuffled'):
        prefix = 'shuffled/'
    else:
        prefix = 'original/'
    outputfile = output_path + prefix + filename + '.csv'
    scheme_letters, root = get_scheme_letters(inputfile)
    # Analysis failed (usually invalid xml because of bugs in SPARSAR).
    if scheme_letters is None:
        return 1
    print(scheme_letters)
    # Create output.
    keys = list(scheme_letters.keys())
    scheme_letter_no = -1
    lines = []
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
            lines.append('{0};{1};{2};{3}\n'.format(scheme_letters[keys[scheme_letter_no]], no, ' '.join(words), ' '.join(phons)))
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    # lines = altered_data_generator.shuffle(lines)
    # print(lines)
    with open(outputfile, 'w+') as output:
        output.write('Rhyme Scheme Letter;Line Number;Lyrics;Phonetic '
                     'Transcription\n')
        output.write(''.join(lines))
    return 0


# Takes list of verses as input.
def add_punctuation(lyrics):
    # Replace newlines with a smiley (because punctuator deletes newlines).
    to_send = '🙂'.join(lyrics)
    cmd = 'curl -d "text=' + to_send + '" http://bark.phon.ioc.ee/punctuator'
    args = shlex.split(cmd)
    # returns output as byte string
    returned_output = subprocess.run(args, check=True,
                                     stdout=subprocess.PIPE).stdout
    # using decode() function to convert byte string to string
    punctuated_lyrics = returned_output.decode("utf-8")
    # Sometimes punctuation is added after newline -> switch it around.
    punctuated_lyrics = re.sub('🙂([.?:!])', '\g<1>🙂', punctuated_lyrics)
    punctuated_lyrics = punctuated_lyrics.split('🙂')
    return punctuated_lyrics


def main():
    start = datetime.now()
    print("Start Time =", start.strftime("%H:%M:%S"))
    # Prepare files for SPARSAR.
    prefixes = ['']
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for prefix in prefixes:
        filename = 'data/' + \
                   prefix + 'ENlyrics_cleaned.json'
        # run_sparsar_for_failed(prefix, filename)
        replace_prohibited_characters(filename)
        # Generate SPARSAR output files.
        run_sparsar_for_files(filename, prefix, start_idx=6400, end_idx=6600)  # Analyzed up to index 6421.
    # Extract useful information from SPARSAR output files to .csv file.
    path = 'sparsar_experiments/outs/'
    output_path = 'sparsar_experiments/rhymes/'
    # output_path = 'sparsar_experiments/line_shuffle_comparison/analyzed_before_shuffle/'
    # path = output_path
    total = 0
    for item in os.listdir(path):
        if isfile(join(path, item)) and item.endswith('_phon.xml'):
            print('Extracting to csv...', item)
            result = extract_rhymes_to_csv(path, item, output_path)
            if result == 0:
                total += 1
    print('Generated', total, '.csv files.')
    end = datetime.now()
    elapsed = end - start
    print("Start Time =", start.strftime("%H:%M:%S"))
    print("End Time =", end.strftime("%H:%M:%S"))
    print("Elapsed Time =", elapsed)


if __name__ == '__main__':
    main()
