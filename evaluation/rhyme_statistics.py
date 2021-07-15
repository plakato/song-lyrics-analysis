# Percentage of line-end words that are not in CMU dictionary.
import argparse


# Songs - array of lines, schemes - array of letters, "-" for no rhyme.
import json
import pprint
from datetime import datetime

from rhyme_detector_v3 import RhymeDetector


def get_scheme_stats(songs, schemes):
    total_songs = 0
    # Number of lines in all songs combined.
    total_lines = 0
    # Number of lines that are part of a rhyme in all songs combined.
    total_rhyming_lines = 0
    # Number of unique rhyme groups in one song, summed for all songs.
    sum_group_count = 0
    # Maximal number of unique rhyme groups per song.
    max_group_count = 0
    # Number of lines in one rhyme group - sum for all songs, max.
    sum_rhyme_group_size = 0
    max_rhyme_group_size = 0

    for song in range(len(songs)):
        total_songs += 1
        groups = {}
        line_count = 0
        for line in range(len(songs[song])):
            if songs[song][line] == '':
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
        if group_count > max_group_count:
            max_group_count = group_count
        for group in groups:
            sum_rhyme_group_size += groups[group]
            if groups[group] > max_rhyme_group_size:
                max_rhyme_group_size = groups[group]
    avg_lines_per_song = total_lines/total_songs
    avg_group_count = sum_group_count / total_songs
    return {'total_songs': total_songs,
            'total_lines': total_lines,
            'total_rhyming_lines': total_rhyming_lines,
            'percentage_of_rhyming_lines': total_rhyming_lines / total_lines,
            'avg_lines_per_song': avg_lines_per_song,
            'avg_group_count_per_100_lines': 100*avg_group_count/avg_lines_per_song,
            'avg_group_count': avg_group_count,
            'max_group_count': max_group_count,
            'avg_group_size': sum_rhyme_group_size/sum_group_count,
            'max_group_size': max_rhyme_group_size}


def get_rating_stats(ratings):
    total_songs = 0
    sum_ratings = 0
    for rating in ratings:
        total_songs += 1
        sum_ratings += rating
    return {'avg_song_rating': sum_ratings/total_songs,
            'median': ratings[int(len(ratings)/2)]}


def get_rhyme_type_stats(rhymes):
    total_rhyming_lines = 0
    perfect_sound_match = 0
    two_syllable = 0
    five_syllable = 0
    eight_syllable = 0
    stress_moved = 0
    perfect_masc = 0
    perfect_fem = 0
    perfect_dact = 0
    imperfect = 0
    forced = 0
    for song in rhymes:
        for i in range(len(song)):
            line = song[i]
            # Skip non-rhyming, empty lines, and first rhyme lines.
            if line['rating'] == 0 or line['rating'] == '-':
                continue
            # If its the second item in the rhyme group, add information for the first as well.
            its_second = song[i+line['rhyme_fellow']]['rhyme_fellow'] == 0
            to_add = 2 if its_second else 1
            total_rhyming_lines += to_add
            if line['stress_moved']:
                stress_moved += to_add
            # Count perfect and imperfect rhymes - they all have matching components.
            equal = line['relevant_components'] == line['relevant_components_rhyme_fellow']
            if equal:
                perfect_sound_match += to_add
                if line['stress_moved']:
                    imperfect += to_add
            else:
                forced += to_add
            if len(line['relevant_components']) == 2:
                two_syllable += to_add
                if not line['stress_moved'] and line['rating'] == 1:
                    perfect_masc += to_add
            elif len(line['relevant_components']) == 5:
                five_syllable += to_add
                if not line['stress_moved'] and line['rating'] == 1:
                    perfect_fem += to_add
            elif len(line['relevant_components']) == 8:
                eight_syllable += to_add
                if not line['stress_moved'] and line['rating'] == 1:
                    perfect_dact += to_add
    return {'total_rhyming_lines': total_rhyming_lines,
            'perfect_sound_match': perfect_sound_match,
            'perc_perfect_sound_match': perfect_sound_match/total_rhyming_lines,
            'two_syllable': two_syllable,
            'five_syllable': five_syllable,
            'eight_syllable': eight_syllable,
            'stress_moved': stress_moved,
            'perfect_masc': perfect_masc,
            'perfect_fem': perfect_fem,
            'perfect_dact': perfect_dact,
            'imperfect': imperfect,
            'forced': forced,
            'perc_two_syllable': two_syllable / total_rhyming_lines,
            'perc_five_syllable': five_syllable / total_rhyming_lines,
            'perc_eight_syllable': eight_syllable / total_rhyming_lines,
            'perc_stress_moved': stress_moved / total_rhyming_lines,
            'perc_perfect_masc': perfect_masc / total_rhyming_lines,
            'perc_perfect_fem': perfect_fem / total_rhyming_lines,
            'perc_perfect_dact': perfect_dact / total_rhyming_lines,
            'perc_imperfect': imperfect / total_rhyming_lines,
            'perc_forced': forced / total_rhyming_lines}


def run_on_data(filename):
    songs = []
    schemes = []
    song_ratings = []
    rhymes = []
    detector = RhymeDetector(matrix_path='data/cooc_iter3.json')
    with open(filename, 'r') as input_data:
        data = json.load(input_data)
    for song in data:
        stats = detector.analyze_lyrics(song['lyrics'])
        songs.append(song['lyrics'])
        schemes.append(stats['scheme'])
        song_ratings.append(stats['song_rating'])
        rhymes.append(stats['ratings'])
    pprint.pprint(get_scheme_stats(songs, schemes))
    pprint.pprint(get_rating_stats(song_ratings))
    pprint.pprint(get_rhyme_type_stats(rhymes))


if __name__ == '__main__':
    genre_files = [
        # '../song_data/data/ENlyrics_final_rock.json',
        #            '../song_data/data/ENlyrics_final_pop.json',
        #            '../song_data/data/ENlyrics_final_r-b.json',
        #            '../song_data/data/ENlyrics_final_country.json',
        #            '../song_data/data/ENlyrics_final_rap.json',
                   '../song_data/data/ENlyrics_final.json']
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()
    for filename in genre_files:
        print(filename)
        print(f'Time: {datetime.now().strftime("%H:%M:%S")}')
        run_on_data(filename)
    # run_on_data('data/train_lyrics0.001.json')