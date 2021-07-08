# Percentage of line-end words that are not in CMU dictionary.
import argparse


# Songs - array of lines, schemes - array of letters, "-" for no rhyme.
import json
import pprint

from rhyme_detector_v3 import RhymeDetector


def scheme_stats(songs, schemes):
    total_songs = 0
    # Number of lines in all songs combined.
    total_lines = 0
    # Number of lines in one song, summed for all songs.
    sum_line_count = 0
    # Number of lines that are part of a rhyme in all songs combined.
    total_rhyming_lines = 0
    # Number of unique rhyme groups in one song, summed for all songs.
    sum_group_count = 0
    # Minimal and maximal number of unique rhyme groups per song.
    min_group_count = 0
    max_group_count = 0
    # Number of lines in one rhyme group - sum for all songs, min, max.
    sum_rhyme_group_size = 0
    min_rhyme_group_size = 0
    max_rhyme_group_size = 0

    for song in range(len(songs)):
        total_songs += 1
        groups = {}
        line_count = 0
        for line in range(len(songs[song])):
            if line == '':
                continue
            total_lines += 1
            line_count += 1
            scheme_letter = schemes[song][line]
            if scheme_letter != "-":
                total_rhyming_lines += 1
                if scheme_letter in groups:
                    groups[scheme_letter] += 1
                else:
                    groups[scheme_letter] = 1
        group_count = len(groups.keys())
        sum_group_count += group_count
        if group_count < min_group_count:
            min_group_count = group_count
        if group_count > max_group_count:
            max_group_count = group_count
        for group in groups:
            sum_rhyme_group_size += groups[group]
            if groups[group] < min_rhyme_group_size:
                min_rhyme_group_size = groups[group]
            if groups[group] > max_rhyme_group_size:
                max_rhyme_group_size = groups[group]

    return {'total_songs': total_songs,
            'total_lines': total_lines,
            'total_rhyming_lines': total_rhyming_lines,
            'percentage_of_rhyming_lines': total_rhyming_lines / total_lines,
            'avg_lines_per_song': sum_line_count/total_songs,
            'avg_group_count': sum_group_count/total_songs,
            'min_group_count': min_group_count,
            'max_group_count': max_group_count,
            'avg_group_size': sum_rhyme_group_size/sum_group_count,
            'min_group_size': min_rhyme_group_size,
            'max_group_size': max_rhyme_group_size}


def run_on_data(filename):
    songs = []
    schemes = []
    detector = RhymeDetector(matrix_path='data/cooc_iter3.json')
    with open(filename, 'r') as input_data:
        data = json.load(input_data)
    for song in data:
        stats = detector.analyze_lyrics(song['lyrics'])
        songs.append(song['lyrics'])
        schemes.append(stats['scheme'])
    pprint.pprint(scheme_stats(songs, schemes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()
    run_on_data('../song_data/data/ENlyrics_final_rock.json')