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

NO_OF_PRECEDING_LINES = 3
IPA_VOWELS = {'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
              'ɪ', 'ʏ', 'ɪ̈', 'ʊ̈', 'ʊ',
              'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
              'ə',
              'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
              'æ', 'ɐ',
              'a', 'ɶ', 'ɑ', 'ɒ'}
dst = panphon.distance.Distance()


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


# Gives next letter given a pattern - alphabetically, after 'z' double 'aa'.
def next_letter_generator():
    for i in itertools.count(1):
        for p in itertools.product(string.ascii_lowercase, repeat=i):
            yield ''.join(p)


# For lyrics, detect standard rhyme types.
def get_rhyme_scheme(lines):
    # Get phonetic transcription and remove punctuation.
    phon_lines = []
    for line in lines:
        # Strip punctuation.
        line = eng_to_ipa.convert(line, keep_punct=False)
        # line = line.translate(str.maketrans('', '', string.punctuation))
        phon_lines.append(line)

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


if __name__ == '__main__':
    poem1 = ['Roses are red','you are tool','please don\'t be mad','be a fool.']
    poem2 = ["Twinkle, twinkle, little star,",
            "How I wonder what you are.",
            "Up above the world so high,",
            "Like a diamond in the sky.",
            "When the blazing sun is gone,",
            "When he nothing shines upon,",
            "Then you show your little light,",
            "Twinkle, twinkle, all the night."]
    lyrics1 = ["We were both young when I first saw you.",
               "I close my eyes and the flashback starts:",
               "I'm standing there",
               "On a balcony in summer air.",
               "See the lights, see the party, the ball gowns,",
               "See you make your way through the crowd,",
               "And say, Hello.",
               "Little did I know...",
               "That you were Romeo, you were throwing pebbles",
               "And my daddy said, Stay away from Juliet.",
               "And I was crying on the staircase",
               "Begging you, Please don't go.",
               "And I said,",
               "Romeo, take me somewhere we can be alone.",
               "I'll be waiting. All there's left to do is run.",
               "You'll be the prince and I'll be the princess.",
               "It's a love story. Baby, just say 'Yes'."]
    # scheme: a, b, c, c, d, e, f, f, g, h, i, j, k, l, m, n, n
    lyrics2 = ["I'm at a party I don't wanna be at",
               "And I don't ever wear a suit and tie, yeah",
               "Wonderin' if I could sneak out the back",
               "Nobody's even lookin' me in my eyes",
               "Can you take my hand?",
               "Finish my drink, say, Shall we dance?",
               "You know I love ya, did I ever tell ya?",
               "You make it better like that",
               "Don't think I fit in at this party",
               "Everyone's got so much to say",
               "I always feel like I'm nobody",
               "Who wants to fit in anyway?",
               "Cause I don't care when I'm with my baby, yeah",
               "All the bad things disappear"]
    # a, a, b, a, c, d, a, c, e, f, e, f, g, g
    if len(sys.argv) > 1 and sys.argv[1] == '-sparsar':
        get_perfect_rhymes_from_sparsar_output()
    else:
        scheme = get_rhyme_scheme(poem1)
        print(scheme)
