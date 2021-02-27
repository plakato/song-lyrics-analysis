import itertools
import os
import string
import sys
import xml.etree.ElementTree as ET
import eng_to_ipa
import cmudict
import panphon.distance

# PARAMETERS
# How many lines should rhyme not repeat to be considered a new rhyme.
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
dst = panphon.distance.Distance()
# Examples of input and output.
poem1 = ['Roses are red', 'you are tool', 'please don\'t be mad', 'be a fool.']
poem2 = ["Twinkle, twinkle, little star,", "How I wonder what you are.", "Up above the world so high,", "Like a diamond in the sky.", "When the blazing sun is gone,", "When he nothing shines upon,", "Then you show your little light,", "Twinkle, twinkle, all the night."]
lyrics1 = ["We were both young when I first saw you.", "I close my eyes and the flashback starts:", "I'm standing there", "On a balcony in summer air.", "See the lights, see the party, the ball gowns,", "See you make your way through the crowd,", "And say, Hello.", "Little did I know...",
           "That you were Romeo, you were throwing pebbles", "And my daddy said, Stay away from Juliet.", "And I was crying on the staircase", "Begging you, Please don't go.", "And I said,", "Romeo, take me somewhere we can be alone.", "I'll be waiting. All there's left to do is run.",
           "You'll be the prince and I'll be the princess.", "It's a love story. Baby, just say 'Yes'."]
# scheme: a, b, c, c, d, e, f, f, g, h, i, j, k, l, m, n, n
lyrics2 = ["I'm at a party I don't wanna be at", "And I don't ever wear a suit and tie, yeah", "Wonderin' if I could sneak out the back", "Nobody's even lookin' me in my eyes", "Can you take my hand?", "Finish my drink, say, Shall we dance?", "You know I love ya, did I ever tell ya?",
           "You make it better like that", "Don't think I fit in at this party", "Everyone's got so much to say", "I always feel like I'm nobody", "Who wants to fit in anyway?", "Cause I don't care when I'm with my baby, yeah", "All the bad things disappear"]
# a, a, b, a, c, d, a, c, e, f, e, f, g, g
lyrics2_syllables_correct = [['wi', 'wər', 'boʊθ', 'jəŋ', 'wɪn', 'aɪ', 'fərst', 'sɔ', 'ju'],
                             ['aɪ', 'kloʊz', 'maɪ', 'aɪz', 'ənd', 'ðə', 'ˈflæʃˌ', 'bæk', 'stɑrts'],
                             ['əm', 'ˈstæn', 'dɪŋ', 'ðɛr'],
                             ['ɔn', 'ə', 'ˈbæl', 'kə', 'ni', 'ɪn', 'ˈsə', 'mər', 'ɛr'],
                             ['si', 'ðə', 'laɪts', 'si', 'ðə', 'ˈpɑr', 'ti', 'ðə', 'bɔl', 'gaʊnz'],
                             ['si', 'ju', 'meɪk', 'jʊr', 'weɪ', 'θru', 'ðə', 'kraʊd'],
                             ['ənd', 'seɪ', 'hɛˈ', 'loʊ'],
                             ['ˈlɪ', 'təl', 'dɪd', 'aɪ', 'noʊ'],
                             ['ðət', 'ju', 'wər', 'ˈroʊ', 'miˌ', 'oʊ', 'ju', 'wər', 'θro', 'ʊɪŋ', 'ˈpɛ', 'bəlz'],
                             ['ənd', 'maɪ', 'ˈdæ', 'di', 'sɛd', 'steɪ', 'əˈ', 'weɪ', 'frəm', 'ˈʤu', 'liˌ', 'ɛt'],
                             ['ənd', 'aɪ', 'wɑz', 'kraɪ', 'ɪŋ', 'ɔn', 'ðə', 'ˈstɛrˌ', 'keɪs'],
                             ['ˈbɛ', 'gɪŋ', 'ju', 'pliz', 'doʊnt', 'goʊ'],
                             ['ənd', 'aɪ', 'sɛd'],
                             ['ˈroʊ', 'miˌ', 'oʊ', 'teɪk', 'mi', 'ˈsəmˌ', 'wɛr', 'wi', 'kən', 'bi', 'əˈ', 'loʊn'],
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


def rhymes(first, second):
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
def get_syllables(line):
    phon_line = convert_to_phonetic([line])[0]
    syllables = []
    current_syllable = []
    vowels_in_syllable = 0
    trailing_consonants = 0
    for a in phon_line:
        # Cases that complete a syllable.
        if a == ' ':
            syllables.append(''.join(current_syllable))
            current_syllable = []
            vowels_in_syllable = 0
            trailing_consonants = 0
        elif vowels_in_syllable > 0 and trailing_consonants > 0 and a in IPA_VOWELS:
            syllables.append(''.join(current_syllable))
            current_syllable = [a]
            vowels_in_syllable = 1
            trailing_consonants = 0
        # Cases that continue a syllable.
        elif a in IPA_VOWELS:
            current_syllable.append(a)
            vowels_in_syllable += 1
        else:
            if vowels_in_syllable > 0:
                current_syllable.append(a)
                trailing_consonants += 1
            else:
                current_syllable.append(a)
    syllables.append(''.join(current_syllable))
    # print(len(syllables), syllables, phon_line)
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
def get_rhyme_scheme(lines):
    phon_lines = convert_to_phonetic(lines)
    print(phon_lines)
    # For each line identify its rhyme buddy or assign a new letter.
    letter_gen = next_letter_generator()
    scheme = [next(letter_gen)]
    for i in range(1, len(lines)):
        # Try if it rhymes with preceding lines.
        rhyme_found = False
        for lines_back in range(1,min(NO_OF_PRECEDING_LINES,i)+1):
            print(f'Checking >{lines[i]}< versus >{lines[i-lines_back]}<')
            they_rhyme, _ = rhymes(phon_lines[i], phon_lines[i-lines_back])
            if they_rhyme:
                rhyme_found = True
                scheme.append(scheme[i-lines_back])
                break
        if not rhyme_found:
            scheme.append(next(letter_gen))
    return scheme


def get_scheme_from_tagger(lyrics):
    rt = RhymeTagger()
    rt.new_model('en', ngram=4, prob_ipa_min=0.95)
    rhymes = rt.tag(lyrics, output_format=3)
    return rhymes


if __name__ == '__main__':
    scheme = ''
    input = lyrics1
    if len(sys.argv) > 1:
        if sys.argv[1] == '-sparsar':
            scheme = get_perfect_rhymes_from_sparsar_output()
        elif sys.argv[1] == '-tagger':
            scheme = get_scheme_from_tagger(poem1)
    else:
        scheme = get_rhyme_scheme(input)
    for i in range(len(input)):
        print(scheme[i], input[i])
    syllables = []
    for line in input:
        syllables.append(get_syllables(line))
    print_syllable_check(syllables, lyrics2_syllables_correct)
