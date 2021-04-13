import re
import sys

from evaluation.constants import NO_OF_PRECEDING_LINES
from evaluation.rhyme_detector_simple import get_syllables_IPA
from evaluation.rhyme_detector_v1 import next_letter_generator, rhymes, get_syllables_ARPA, get_pronunciation_for_word

NOT_AVAILABLE = 'NOT_AVAILABLE'


def get_syllable_count_and_pronunciations(lyrics):
    syll_counts = []
    pronunciations = []
    for line in lyrics:
        count = 0
        words = line.split(' ')
        line_pron = []
        for word in words:
            sylls = None
            # Tries to get syllables using syllabify library.
            pron = get_pronunciation_for_word(word)
            if pron:
                sylls = get_syllables_ARPA(pron[0])
                line_pron.append(' '.join(pron[0]))
            if not sylls:
                # Estimates number of syllables as number of vowel groups in the word.
                vowel_group_patt = re.compile('[aeiouy]+')
                sylls = re.findall(vowel_group_patt, word)
                line_pron.append(word)
                print(f'Haven\'t found syllable count for word: {word}, estimating {len(sylls)} syllables.')
            count += len(sylls)
        syll_counts.append(count)
        pronunciations.append(line_pron)
    print('LINE SYLLABLE COUNTS:')
    for i in range(len(lines)):
        print(syll_counts[i], lines[i])
    return syll_counts, pronunciations


# Prepare lines by removing punctuation and empty lines.
def preprocess_lyrics(lyrics):
    lines = []
    for line in lyrics:
        line = re.sub(r"-", " ", line)
        line = re.sub(r"’", "'", line)
        line = re.sub(r"[^\w\d'\s]+", '', line)
        line = ' '.join(line.split())
        if line != '':
            lines.append(line)
    return lines


# For lyrics, detect standard rhyme types.
def get_rhyme_scheme_and_rating(lines):
    rating_for_letter = {}
    pronunciations = ['']*len(lines)
    # For each line identify its rhyme buddy or assign a new letter.
    letter_gen = next_letter_generator()
    scheme = [next(letter_gen)]
    ratings = [0]*len(lines)
    # We ignore the rating of first line of the rhymed pair. Otherwise we would count each rating twice.
    ignored_ratings = [False]*len(lines)
    for i in range(1, len(lines)):
        # Try if it rhymes with preceding lines.
        rhyme_found = False
        for lines_back in range(1, min(NO_OF_PRECEDING_LINES, i)+1):
            print(f'Checking >{lines[i]}< versus >{lines[i-lines_back]}<')
            they_rhyme, stats = rhymes(lines[i], lines[i-lines_back])
            # We need to add the pronunciation for the first line separately, because it was skipped.
            if i-lines_back == 0:
                pronunciations[0] = stats['pronunciation']['pronunciation2']
            # If we're missing second pronunciation, don't use the first unless it's empty.
            if stats['pronunciation']['pronunciation1'] and pronunciations[i] == '':
                pronunciations[i] = stats['pronunciation']['pronunciation1']
            if they_rhyme:
                rhyme_found = True
                letter = scheme[i-lines_back]
                scheme.append(letter)
                ratings[i] = stats['rating']
                # Replace by the pronunciation used.
                pronunciations[i] = stats['pronunciation']['pronunciation1']
                pronunciations[i-lines_back] = stats['pronunciation']['pronunciation2']
                if ratings[i-lines_back] == 0:
                    ratings[i-lines_back] = stats['rating']
                    ignored_ratings[i-lines_back] = True
                # There already is a rating but it's worse - replace it with new rating
                # TODO think about this case...AABB vs AcAcA
                elif stats['rating'] > ratings[i-lines_back] > 0:
                    ignored_ratings[i - lines_back] = True
                    ratings[i - lines_back] = stats['rating']
                break
        if not rhyme_found:
            scheme.append(next(letter_gen))
    print("Ratings for individual rhyme scheme letters:", rating_for_letter)
    # Calculate overall rating.
    sum_of_line_ratings = 0
    n_valid_lines = 0
    for i in range(len(ratings)):
        # First lines in rhyme pair carry the rating in list. We ignore them to count each rhyme pair only once.
        if ignored_ratings[i]:
            continue
        sum_of_line_ratings += ratings[i]
        n_valid_lines += 1
    song_rating = sum_of_line_ratings/n_valid_lines
    # Print the results.
    for i in range(len(lines)):
        if ignored_ratings[i]:
            rating_value = NOT_AVAILABLE
        else:
            rating_value = str(ratings[i]) + ' '*10
        if not pronunciations[i]:
            pron = 'PRONUNCIATION_NOT_FOUND'
        else:
            # Convert to more readable form.
            pron = [' '.join(' '.join(letters) for letters in triplet) for triplet in pronunciations[i]]
            # Remove possible meter.
            pron = [re.sub('[012]', '', syll) for syll in pron]
        print(f'{scheme[i]:<2}', f'{rating_value[:13]:<13}', lines[i], pron)
    print("RATING:", song_rating)
    return scheme, song_rating, lines


if __name__ == '__main__':
    if len(sys.argv) < 2 or not sys.argv[1] == '-test_lyrics':
        print('Please enter the input for analysis using --test_lyrics file_path')
        exit(11)
    with open(sys.argv[2]) as input_file:
        input = input_file.read().splitlines()
    lines = [line for line in input if line]
    lines = preprocess_lyrics(lines)
    syll_counts, _ = get_syllable_count_and_pronunciations(lines)
    scheme, rating, lines = get_rhyme_scheme_and_rating(lines)
