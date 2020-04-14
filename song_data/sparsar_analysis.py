# -*- coding: utf-8 -*-
import json
import os
import subprocess


filename = 'data/100lyrics_cleaned.json'

def create_sparsar_input_file_from_song(song, output_filename):
    with open(output_filename, 'w+') as output:
        output.write(song['title'] + '\n')
        output.write('by ' + song['artist'] + '.\n\n')
        for i in range(len(song['lyrics'])):
            line = song['lyrics'][i] # .replace('-', '-')
            # line = line.replace(u'–', '-')
            # line = line.replace('\'', '\'')
            # line = line.replace('\'', '\'')
            output.write(line.encode('utf-8') + '\n')

def main():
    os.environ["PYTHONIOENCODING"] = "utf-8"
    with open(filename) as input:
        data = json.load(input)
        os.chdir("./sparsar_experiments")
        for song in data:
            if song['lang'] == 'ENGLISH':
                song_file = song['title'] + '.txt'
                print(song_file)
                create_sparsar_input_file_from_song(song, song_file)
                os.system("./sparsar_man loadallouts -- \"" + song_file + "\"")

if __name__ == '__main__':
    main()
