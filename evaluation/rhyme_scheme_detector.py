import itertools
import math
import os
import re
import string
import sys
import xml.etree.ElementTree as ET
import eng_to_ipa
import cmudict
import panphon.distance
from syllabify import syllabify

# PARAMETERS
# How many lines should rhyme not repeat to be considered a new rhyme.
import pronouncing
from rhymetagger import RhymeTagger
from torch.utils.hipify.hipify_python import bcolors

NO_OF_PRECEDING_LINES = 3
IPA_VOWELS = {'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
              'ɪ', 'ʏ', 'ɪ̈', 'ʊ̈', 'ʊ',
              'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
              'ə',
              'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
              'æ', 'ɐ',
              'a', 'ɶ', 'ɑ', 'ɒ'}
ARPA_VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
               'EH', 'ER', 'EY',
               'IH', 'IX', 'IY',
               'OW', 'OY',
               'UH', 'UW', 'UX'}
dst = panphon.distance.Distance()
dict = cmudict.dict()
# Examples of input and output.
poem1 = ['Roses are red', 'you are too', "please don't be mad", 'be a fool.']
poem2 = ["Twinkle, twinkle, little star,", "How I wonder what you are.", "Up above the world so high,", "Like a diamond in the sky.", "When the blazing sun is gone,", "When he nothing shines upon,", "Then you show your little light,", "Twinkle, twinkle, all the night."]
lyrics1 = ["We were both young when I first saw you.", "I close my eyes and the flashback starts:", "I'm standing there", "On a balcony in summer air.", "See the lights, see the party, the ball gowns,", "See you make your way through the crowd,", "And say, Hello.", "Little did I know...",
           "That you were Romeo, you were throwing pebbles", "And my daddy said, Stay away from Juliet.", "And I was crying on the staircase", "Begging you, Please don't go.", "And I said,", "Romeo, take me somewhere we can be alone.", "I'll be waiting. All there's left to do is run.",
           "You'll be the prince and I'll be the princess.", "It's a love story. Baby, just say 'Yes'."]
# scheme: a, b, c, c, d, e, f, f, g, h, i, j, k, l, m, n, n
lyrics2 = ["I'm at a party I don't wanna be at", "And I don't ever wear a suit and tie, yeah", "Wonderin' if I could sneak out the back", "Nobody's even lookin' me in my eyes", "Can you take my hand?", "Finish my drink, say, Shall we dance?", "You know I love ya, did I ever tell ya?",
           "You make it better like that", "Don't think I fit in at this party", "Everyone's got so much to say", "I always feel like I'm nobody", "Who wants to fit in anyway?", "Cause I don't care when I'm with my baby, yeah", "All the bad things disappear"]
# a, a, b, a, c, d, a, c, e, f, e, f, g, g
lyrics2_syllables_correct = [['wi', 'wər', 'boʊθ', 'jəŋ', 'wɪn', 'aɪ', 'fərst', 'sɔ', 'ju'],
                             ['aɪ', 'kloʊz', 'maɪ', 'aɪz', 'ənd', 'ðə', 'ˈflæʃ', 'ˌbæk', 'stɑrts'],
                             ['əm', 'ˈstæn', 'dɪŋ', 'ðɛr'],
                             ['ɔn', 'ə', 'ˈbæl', 'kə', 'ni', 'ɪn', 'ˈsə', 'mər', 'ɛr'],
                             ['si', 'ðə', 'laɪts', 'si', 'ðə', 'ˈpɑr', 'ti', 'ðə', 'bɔl', 'gaʊnz'],
                             ['si', 'ju', 'meɪk', 'jʊr', 'weɪ', 'θru', 'ðə', 'kraʊd'],
                             ['ənd', 'seɪ', 'hɛˈ', 'loʊ'],
                             ['ˈlɪ', 'təl', 'dɪd', 'aɪ', 'noʊ'],
                             ['ðət', 'ju', 'wər', 'ˈroʊ', 'mi', 'ˌoʊ', 'ju', 'wər', 'θro', 'ʊɪŋ', 'ˈpɛ', 'bəlz'],
                             ['ənd', 'maɪ', 'ˈdæ', 'di', 'sɛd', 'steɪ', 'əˈ', 'weɪ', 'frəm', 'ˈʤu', 'li', 'ˌɛt'],
                             ['ənd', 'aɪ', 'wɑz', 'kraɪ', 'ɪŋ', 'ɔn', 'ðə', 'ˈstɛr', 'ˌkeɪs'],
                             ['ˈbɛ', 'gɪŋ', 'ju', 'pliz', 'doʊnt', 'goʊ'],
                             ['ənd', 'aɪ', 'sɛd'],
                             ['ˈroʊ', 'mi', 'ˌoʊ', 'teɪk', 'mi', 'ˈsəm', 'ˌwɛr', 'wi', 'kən', 'bi', 'əˈ', 'loʊn'],
                             ['aɪl', 'bi', 'ˈweɪ', 'tɪŋ', 'ɔl', 'ðɛrz', 'lɛft', 'tɪ', 'du', 'ɪz', 'rən'],
                             ['jul', 'bi', 'ðə', 'prɪns', 'ənd', 'aɪl', 'bi', 'ðə', 'ˈprɪn', 'sɛs'],
                             ['ɪts', 'ə', 'ləv', 'ˈstɔ', 'ri', 'ˈbeɪ', 'bi', 'ʤɪst', 'seɪ', 'jɛs']]


def load_lines_from_sparsar_output(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    words = []
    phons = []
    for stanza in root[0]:
        for line in stanza[1]:
            words_line = []
            phons_line = []
            # print(line.attrib['line_syllables'].split(']'))
            for word in line.attrib['line_syllables'].split(']'):
                if word == '':
                    continue
                word = word.replace('[', '').replace(',', '')
                parts = word.split('/')
                words_line.append(parts[0])
                phons_line.append(parts[1].split(','))
            words.append(words_line)
            phons.append(phons_line)
    return words, phons


def find_perfect_rhymes(lines):
    letters = list(string.ascii_letters)
    mapping = {}
    scheme = []
    for line in lines:
        last2phons = '_'.join(line[-1][-1].split('_')[-2:])
        if last2phons not in mapping:
            letter = letters.pop(0)
            mapping[last2phons] = letter
        else:
            letter = mapping[last2phons]
        scheme.append(letter)
    return scheme


def get_perfect_rhymes_from_sparsar_output():
    words, phons = load_lines_from_sparsar_output('sparsar_experiments/outs/\'Ask Me '
                              'Lyrics.txt\'_phon.xml')
    scheme = find_perfect_rhymes(phons)
    i = 0
    for i in range(len(scheme)):
        print('{0} {1}'.format(scheme[i], ' '.join(words[i])))
    return scheme


def get_pronunciation_for_word(word):
    # Strip punctuation except apostrophes.
    word = re.sub(r"[^\w\d'\s]+",'',word)
    # Convert to all lower-case.
    word = word.lower()
    pronunciations = dict.get(word)
    if pronunciations is None:
        print(f"Haven't found pronunciation for {word}.")
        # todo fallback when not in dictionary
    return pronunciations


# Returned format is a list of triples - each triplet is one syllable, split to parts by pattern CVC (conconants, vowels, consonants).
def get_syllables_ARPA(word):
    syllables = syllabify.syllabify(word)
    return syllables


def get_pronunciations_for_n_syllables(line, n):
    words = [word for word in line.split(' ') if word]
    result = None
    for word in reversed(words):
        pronunciations = get_pronunciation_for_word(word)
        if len(pronunciations) == 0:
            return ''
        syllabifications = [get_syllables_ARPA(pron) for pron in pronunciations]
        # Append to result.
        if not result:
            result = syllabifications
        else:
            # We have a result already, we have to join each with each.
            new_result = []
            for sylab in syllabifications:
                for r in result:
                    new_result.append(sylab+r)
            result = new_result
        # If we have n syllables in all syllabifications we're done.
        reached_n = True
        for r in result:
            if len(r) < n:
                reached_n = False
                break
        if reached_n:
            break
    # Shorten each syllabification to n syllables.
    for i in range(len(result)):
        if len(result[i]) > n:
            result[i] = result[i][-4:]
    return result


def extract_syllable_stresses(syllables):
    stresses = []
    pron = []
    for con1, vow, con2 in syllables:
        s = re.findall('[012]', ' '.join(vow))
        if len(s) > 1:
            print(f'BIG BIG ERROR: More than one stress in one syllable. \nSyllable vowels: {vow}\nsequence of syllables {syllables}')
            exit(1)
        if len(s) == 0:
            s = ['0']
        stresses.append(int(s[0]))
        pron.append((con1, [re.sub('[012]', '', v) for v in vow], con2))
    return stresses, pron


def pronunciations_rhyme(pron1, pron2, meter1, meter2):
    # Set variables.
    rhyme_found = False
    perfect_rhyme = None
    identity = False
    imperfect = False
    syllabic = False
    weak = False
    forced = False
    assonance = 0
    conconance = 0
    # Find non-rhymes first to save time.
    if pron1[-1] != pron2[-1]:
        pass
    elif pron1[-2] != pron2[-2]:
        if meter1[-1] == '1' and meter2[-1] == '1':
            perfect_rhyme = 'masculine'
        elif meter1[-1] == '0' and meter2[-1] == '0':
            syllabic = True
        else:
            imperfect = True
    elif pron1[-3] != pron2[-3]:
        if meter1[-2:] == ['1', '0'] and meter2[-2:] == ['1', '0']:
            perfect_rhyme = 'feminine'
        elif meter1[-2:] == ['0', '0'] and meter2[-2:] == ['0', '0']:
            syllabic = True
        else:
            imperfect = True
    elif pron1[-3] != pron2[-3]:
        if meter1[-3:] == ['1', '0', '0'] and meter2[-3:] == ['1', '0', '0']:
            perfect_rhyme = 'dactylic'
        else:
            imperfect = True
    else:
        identity = True
    if perfect_rhyme or identity or imperfect or weak or forced:
        rhyme_found = True
    return rhyme_found, {'perfect': perfect_rhyme,
                         'identity': identity,
                         'impefect': imperfect,
                         'weak': weak,
                         'forced': forced,
                         'syllabic': syllabic,
                         'assosnance': assonance,
                         'consonance': conconance}


# For a pair of phonetic transcription assign value whether the corresponding pair is
#   1 - exactly the same
#   0.5 - sounds similar
#   0 - absent on at least one side
#   -1 - totally different
def evaluate_similarity(A, B):
    n = min(len(A), len(B))
    result = [(-1, -1, -1)]*n
    # Traverse backwards in case there is an uneven number of syllables.
    for i in range(1, n+1):
        Acon1, Avow, Acon2 = A[-i]
        Bcon1, Bvow, Bcon2 = B[-i]
        con1 = evaluate_similarity_for_phoneme_group(Acon1, Bcon1)
        vow = evaluate_similarity_for_phoneme_group(Avow, Bvow)
        con2 = evaluate_similarity_for_phoneme_group(Acon2, Bcon2)
        result[-i] = (con1, vow, con2)
    return result


def evaluate_similarity_for_phoneme_group(A, B):
    # Empty group does not indicate similarity in sound, only in structure.
    if A == [] and B == []:
        return 0
    if A == [] or B == []:
        return -0.5
    if A == B:
        return 1
    # One is a subset of another.
    if set(A) <= set(B) or set(B) <= set(A):
        return 0.5
    # Check whether multi-letter groups don't share a letter.
    if len(A) > 1 or len(B) > 1:
        if len(B) < len(A):
            shorter = B
            longer = A
        else:
            shorter = A
            longer = B
        for i in range(len(longer)-len(shorter) +1):
            if shorter[0] == longer[0+i] or shorter[-1] == longer[-1-i]:
                return 0.5
    return -1


# Returns meter and similarity statistics for each possible pronunciation combination.
def get_stats_for_verse_pair(first, second):
    statistics = []
    # Get pronunciations for line-end syllables.
    end_pronunciations1 = get_pronunciations_for_n_syllables(first, 4)
    end_pronunciations2 = get_pronunciations_for_n_syllables(second, 4)
    # Look at each pair of pronunciations.
    for end_pron1 in end_pronunciations1:
        stresses1, pron1 = extract_syllable_stresses(end_pron1)
        for end_pron2 in end_pronunciations2:
            stresses2, pron2 = extract_syllable_stresses(end_pron2)
            similarity = evaluate_similarity(pron1, pron2)
            statistics.append({'pronunciation1': pron1, 'pronunciation2': pron2, 'meter1': stresses1, 'meter2': stresses2, 'similarity': similarity})
    return statistics


def get_rhyme_rating(stats):
    total_rating = 0
    min_syll_rating = 0.6
    # How much does each group contribute to rhyme.
    c1_weight = 1
    c2_weight = 2
    v_weight = 4
    identity_penalty = 0.8
    total = c1_weight + c2_weight + v_weight
    syll_considered = 1

    def rate_syllable(meter1, meter2, c1, v, c2):
        rating = 0
        # Consider secondary stress as primary.
        meter1 = meter1 if meter1 <= 1 else 1
        meter2 = meter2 if meter2 <= 1 else 1
        meter_rating = 1
        if meter1 != meter2:
            meter_rating = 0.8
        elif meter1 == 0:
            meter_rating = 0.9
        # First get rid of non-rhymes.
        if c1 <= 0 and c2 <= 0 and v <= 0:
            return 0
        # Give penalty for identity.
        if c1 in [0,1] and c2 == 1 and v == 1 and meter1 == 1 and meter2 == 1:
            return identity_penalty
        # Modify rating depending on similarities and their positions.
        if v >= 0:
            rating += v_weight * v
            # Lower rating if last consonant doesn't match.
            if c2 < 0:
                rating += c2/10
        # If last consonant is absent, it's technically a match.
        if c2 >= 0:
            rating += c2_weight*(c2 if c2 != 0 else 1)
        # If first consonant matches or the condition for perfect rhyme is fulfilled.
        if c1 >= 0 or (c2 >= 0 and v >= 0):
            rating += c1_weight
        elif c1 < 0:
            rating += c1/10
        return meter_rating*rating/total

    # Look at last syllable.
    last_syll_rating = rate_syllable(stats['meter1'][-1], stats['meter2'][-1], *stats['similarity'][-1])
    # Look at second-to-last syllable.
    if len(stats['meter1']) > 1 and len(stats['meter1']) > 1:
        sectolast_syll_rating = rate_syllable(stats['meter1'][-2], stats['meter2'][-2], *stats['similarity'][-2])
        # We ignore this syllable if we have low rating and we've already seen primary stresses.
        # It means rhyme doesn't extend to this syllable.
        if sectolast_syll_rating < min_syll_rating and \
            (stats['meter1'][-2] >= 1 or stats['meter1'][-1] >= 1) and \
            (stats['meter2'][-2] >= 1 or stats['meter2'][-1] >= 1):
            return last_syll_rating
        else:
            total_rating = sectolast_syll_rating + last_syll_rating
            syll_considered += 1
    # Look at third-to-last syllable.
    if len(stats['meter1']) > 2 and len(stats['meter1']) > 2:
        thirdtolast_syll_rating = rate_syllable(stats['meter1'][-3], stats['meter2'][-3], *stats['similarity'][-3])
        # We ignore this syllable if we have low rating.
        if thirdtolast_syll_rating < min_syll_rating:
            return total_rating/syll_considered
        else:
            total_rating += thirdtolast_syll_rating
            syll_considered += 1
    # Look at fourth-to-last syllable.
    if len(stats['meter1']) > 3 and len(stats['meter1']) > 3:
        fourthtolast_syll_rating = rate_syllable(stats['meter1'][-4], stats['meter2'][-4], *stats['similarity'][-4])
        # We ignore this syllable if we have low rating.
        if fourthtolast_syll_rating < min_syll_rating:
            return total_rating / syll_considered
        else:
            return (total_rating + fourthtolast_syll_rating)/(syll_considered + 1)
    return total_rating/syll_considered


# Evaluates a pair of lines and returns the most rhyming pronounciations with their rating.
def rhymes(first, second):
    rhyme_found = False
    statistics = get_stats_for_verse_pair(first, second)
    highest_rating = -1
    highest_rated_combo = None
    for combo in statistics:
        rating = get_rhyme_rating(combo)
        if rating > highest_rating:
            highest_rating = rating
            highest_rated_combo = combo
    if highest_rating > 0.7:
        rhyme_found = True
    return rhyme_found, {'rating': highest_rating, 'pronunciation': highest_rated_combo}


# First attempt to create rhyme detector. Uses IPA translation and checks for identity or similarity using Dogol distance.
def simple_rhyme_detector(first, second):
    first = convert_to_phonetic([first])[0]
    second = convert_to_phonetic([second])[0]
    # Find matching phonemes by transversing the line backwards.
    n_perfect_match = 0
    n_close_match = 0
    skipped_phons = 0
    rhyme_found = False
    i1 = len(first) -1
    i2 = len(second) -1
    while i1 >= 0 and i2 >= 0:
        # Matching phonemes.
        if first[i1] == second[i2]:
            if first[i1] in IPA_VOWELS:
                rhyme_found = True
            n_perfect_match += 1
            i1 -= 1
            i2 -= 1
        # Related phonemes.
        elif dst.dogol_prime_distance(first[i1], second[i2]) < 1:
            if first[i1] in IPA_VOWELS:
                rhyme_found = True
            n_close_match +=1
            i1 -= 1
            i2 -= 1
        # Space -> skip.
        elif first[i1] == ' ':
            i1 -= 1
        elif second[i2] == ' ':
            i2 -= 1
        # Different classes -> skip.
        elif first[i1] not in IPA_VOWELS and second[i2] in IPA_VOWELS:
            skipped_phons += 1
            i1 -= 1
        elif first[i1] in IPA_VOWELS and second[i2] not in IPA_VOWELS:
            skipped_phons += 1
            i2 -= 1
        # Totally different phonemes -> end.
        else:
            break
    return rhyme_found, {'perfect_match': n_perfect_match, 'close_match': n_close_match, 'skipped_phonemes': skipped_phons}


# Syllables based on number of vowels in phonetic translation.
def get_syllables_IPA(line):
    phon_line = convert_to_phonetic([line])[0]
    syllables = []
    current_syllable = []
    vowels_in_syllable = 0
    trailing_consonants = 0
    # We go backwards because it's easier.
    for word in phon_line.split(' '):
        i = 0
        while i < len(word):
            # Let's create a syllable.
            while i < len(word) and word[i] not in IPA_VOWELS:
                current_syllable.append(word[i])
                i += 1
            # Max two vowels per syllable.
            vowels = 0
            while i < len(word) and word[i] in IPA_VOWELS:
                current_syllable.append(word[i])
                vowels += 1
                i += 1
                if vowels == 2:
                    break
            # Find next vowel and take half of the consonants in between.
            cons_between = 0
            explorer_i = i
            while explorer_i < len(word) and word[explorer_i] not in IPA_VOWELS:
                explorer_i += 1
                cons_between += 1
            n_consonants_to_assign = int(cons_between/2) if explorer_i < len(word) else cons_between
            # Append assigned consonants.
            while n_consonants_to_assign > 0:
                current_syllable.append(word[i])
                i += 1
                n_consonants_to_assign -= 1
            syllables.append(''.join(current_syllable))
            current_syllable = []
    return syllables


def print_syllable_check(computed, correct):
    for i in range(len(computed)):
        com_i = 0
        corr_i = 0
        while com_i < len(computed[i]):
            if computed[i][com_i] == correct[i][corr_i]:
                print(computed[i][com_i], end=' ')
                com_i += 1
                corr_i += 1
            else:
                do_print = True
                while computed[i][com_i] != correct[i][corr_i]:
                    if do_print:
                        print(f'{bcolors.WARNING}{computed[i][com_i]}{bcolors.ENDC}', end=' ')
                    do_print = True
                    if len(computed[i][com_i]) < len(correct[i][corr_i]):
                        correct[i][corr_i] = correct[i][corr_i].replace(computed[i][com_i], '', 1)
                        com_i += 1
                    else:
                        computed[i][com_i] = computed[i][com_i].replace(correct[i][corr_i], '', 1)
                        corr_i += 1
                        do_print = False
                    if com_i > len(computed[i]) - 1:
                        break
        print()


# Gives next letter given a pattern - alphabetically, after 'z' double 'aa'.
def next_letter_generator():
    for i in itertools.count(1):
        for p in itertools.product(string.ascii_lowercase, repeat=i):
            yield ''.join(p)


# Get phonetic transcription and remove punctuation.
def convert_to_phonetic(lines):
    phon_lines = []
    for line in lines:
        # Strip punctuation.
        line = eng_to_ipa.convert(line, keep_punct=False)
        # line = line.translate(str.maketrans('', '', string.punctuation))
        phon_lines.append(line)
    return phon_lines


# For lyrics, detect standard rhyme types.
def get_rhyme_scheme_and_rating(lyrics):
    sum_of_rhyme_ratings = 0
    n_rhymes = 0
    rating_for_letter = {}
    # Prepare lines by removing punctuation and empty lines.
    lines = []
    for line in lyrics:
        line = re.sub(r"-", " ", line)
        line = re.sub(r"’", "'", line)
        line = re.sub(r"[^\w\d'\s]+",'',line)
        line = ' '.join(line.split())
        if line != '':
            lines.append(line)
    # For each line identify its rhyme buddy or assign a new letter.
    letter_gen = next_letter_generator()
    scheme = [next(letter_gen)]
    ratings = ['NO_RHYME']*len(lines)
    for i in range(1, len(lines)):
        # Try if it rhymes with preceding lines.
        rhyme_found = False
        for lines_back in range(1, min(NO_OF_PRECEDING_LINES, i)+1):
            print(f'Checking >{lines[i]}< versus >{lines[i-lines_back]}<')
            they_rhyme, stats = rhymes(lines[i], lines[i-lines_back])
            if they_rhyme:
                rhyme_found = True
                letter = scheme[i-lines_back]
                scheme.append(letter)
                rating_for_letter[letter] = stats['rating']
                sum_of_rhyme_ratings += stats['rating']
                n_rhymes += 1
                ratings[i] = stats['rating']
                if ratings[i-lines_back] == 'NO_RHYME':
                    ratings[i-lines_back] = 'NOT_AVAILABLE'
                break
        if not rhyme_found:
            scheme.append(next(letter_gen))
    print("Ratings for individual rhyme scheme letters:", rating_for_letter)
    # Calculate number of non-rhyming lines.
    nonrhyming_letters_unique = set()
    rhyming_letters_unique = set()
    for l in scheme:
        if l in nonrhyming_letters_unique:
            rhyming_letters_unique.add(l)
        else:
            nonrhyming_letters_unique.add(l)
    for letter in rhyming_letters_unique:
        nonrhyming_letters_unique.remove(letter)
    no_nonrhyming_lines = len(nonrhyming_letters_unique)
    unique_letters = len(set(scheme))
    no_rhyming_lines = len(lines) - no_nonrhyming_lines
    # Calculate overall rating.
    avg_rhyme_rating = sum_of_rhyme_ratings/n_rhymes
    rhymed_lines_ratio = no_rhyming_lines/len(lines) # math.log(no_rhyming_lines, len(lines))
    song_rating = avg_rhyme_rating * rhymed_lines_ratio
    # Print the results.
    for i in range(len(lines)):
        print(scheme[i], lines[i], ratings[i])
    print("RATING:", song_rating)
    return scheme, song_rating, lines


def get_scheme_from_tagger(lyrics):
    rt = RhymeTagger()
    rt.new_model('en', ngram=4, prob_ipa_min=0.95)
    rhymes = rt.tag(lyrics, output_format=3)
    return rhymes


if __name__ == '__main__':
    scheme = ''
    input = poem1
    if len(sys.argv) == 1:
        if sys.argv[1] == '-sparsar':
            scheme = get_perfect_rhymes_from_sparsar_output()
        elif sys.argv[1] == '-tagger':
            scheme = get_scheme_from_tagger(poem1)
    else:
        if sys.argv[1] == '-test_lyrics':
            with open('test_lyrics/'+sys.argv[2]) as input_file:
                input = input_file.read().splitlines()
        lines = [line for line in input if line]
        scheme, rating, lines = get_rhyme_scheme_and_rating(lines)
