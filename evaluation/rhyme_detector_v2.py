import re
import sys

from evaluation.constants import NO_OF_PRECEDING_LINES
import evaluation.rhyme_detector_v1 as rd1

NOT_AVAILABLE = 'X'


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
            pron = rd1.get_pronunciation_for_word(word)
            if pron:
                sylls = rd1.get_syllables_ARPA(pron[0])
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


def get_rhyme_rating(stats):
    ratings = []
    # How much does each group contribute to rhyme.
    c1_weight = 1
    c2_weight = 2
    v_weight = 4
    identity_penalty = 0.8
    total = c1_weight + c2_weight + v_weight
    syll_considered = 0
    # Where to stop analysis - if we've seen both stresses (looking backwards).
    stop = max(min(filter(lambda i: stats['meter1'][-i] >= 1, range(1, len(stats['meter1'])+1))),
               min(filter(lambda i: stats['meter2'][-i] >= 1, range(1, len(stats['meter2'])+1))))

    def rate_syllable(meter1, meter2, c1, v, c2):
        rating = 0
        # Consider secondary stress as primary.
        meter1 = meter1 if meter1 <= 1 else 1
        meter2 = meter2 if meter2 <= 1 else 1
        # Both stressed have the best rating.
        # But if it's identity, give it full rating as well, identity with different stress doesn't make sense.
        if (meter1 == 1 and meter2 == 1) or (c1 == 1 and v == 1 and c2 == 1):
            meter_rating = 1
        elif meter1 != meter2:
            meter_rating = 0.8
        elif meter1 == 0:
            meter_rating = 0.9
        # First get rid of non-rhymes.
        if c1 <= 0 and c2 <= 0 and v <= 0:
            return 0
        # Give penalty for identity.
        if c1 in [0, 1] and c2 in [0, 1] and v == 1 and meter1 == 1 and meter2 == 1:
            return identity_penalty
        # Modify rating depending on similarities and their positions.
        if v >= 0:
            rating += v_weight * v
        # If last consonant is absent, it's technically a match.
        if c2 >= 0:
            rating += c2_weight*(c2 if c2 != 0 else 1)
        # If first consonant matches or the condition for perfect rhyme is fulfilled.
        if c1 >= 0 or (c2 >= 0 and v >= 0):
            rating += c1_weight
        elif c1 < 0:
            rating += c1/10
        return meter_rating*rating/total

    # Look at syllables backwards one by one until stop index.
    for i in range(1, min(len(stats['meter1']), len(stats['meter2'])) + 1):
        syll_rating = rate_syllable(stats['meter1'][-i], stats['meter2'][-i], *stats['similarity'][-i])
        # Special case for multisyllable perfect rhyme.
        if i > 1 and syll_rating == 1.0 and stats['meter1'][-i] >= 1 and stats['meter2'][-i] >= 1 and \
                all(r == identity_penalty for r in ratings):  # all previous ratings were identities
            ratings = [1.0]*len(ratings)    # We give it a perfect rating.
        ratings = [syll_rating] + ratings
        syll_considered += 1
        if stop == i:
            return sum(ratings)/syll_considered
    return sum(ratings)/syll_considered


# Evaluates a pair of lines and returns the most rhyming pronunciations with their rating.
def rhymes(first, second):
    rhyme_found = False
    statistics = rd1.get_stats_for_verse_pair(first, second)
    if len(statistics) == 1 and not statistics[0]['similarity']:
        return False, {'rating': 0, 'pronunciation': statistics[0]}
    highest_rating = -1
    highest_rated_combo = None
    for combo in statistics:
        rating = get_rhyme_rating(combo)
        if rating > highest_rating:
            highest_rating = rating
            highest_rated_combo = combo
    if highest_rating >= 0.7:
        rhyme_found = True
    return rhyme_found, {'rating': highest_rating, 'pronunciation': highest_rated_combo}


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
    syll_counts, _ = get_syllable_count_and_pronunciations(lines)
    rating_for_letter = {}
    pronunciations = ['']*len(lines)
    # For each line identify its rhyme buddy or assign a new letter.
    letter_gen = rd1.next_letter_generator()
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
                elif stats['rating'] > ratings[i-lines_back] > 0:
                    # Careful - only replace if previous line has something else to rhyme with - otherwise keep both.
                    for preceding in range(1, NO_OF_PRECEDING_LINES+1):
                        index = i-lines_back-preceding
                        if index >= 0 and scheme[index] == letter and not ignored_ratings[index]:
                            ignored_ratings[i - lines_back] = True
                            ratings[i - lines_back] = stats['rating']
                            break
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
        print(f'{scheme[i]:<2}', f'{rating_value[:3]:<3}', f'{syll_counts[i]:<2}', lines[i], pron)
    print("RATING:", song_rating)
    return scheme, song_rating, lines


if __name__ == '__main__':
    if len(sys.argv) < 2 or not sys.argv[1] == '-data':
        print('Please enter the input for analysis using --data file_path')
        exit(11)
    with open(sys.argv[2]) as input_file:
        input = input_file.read().splitlines()
    lines = [line for line in input if line]
    lines = preprocess_lyrics(lines)
    scheme, rating, lines = get_rhyme_scheme_and_rating(lines)
