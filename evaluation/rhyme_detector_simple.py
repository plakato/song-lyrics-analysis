import eng_to_ipa

from evaluation.constants import dst
from evaluation.constants import IPA_VOWELS


# Get phonetic transcription and remove punctuation.
def convert_to_phonetic(lines):
    phon_lines = []
    for line in lines:
        # Strip punctuation.
        line = eng_to_ipa.convert(line, keep_punct=False)
        phon_lines.append(line)
    return phon_lines


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
