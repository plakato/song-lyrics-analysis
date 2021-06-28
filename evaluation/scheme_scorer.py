import argparse

import sklearn.metrics

from rhyme_detector_v1 import next_letter_generator
from universal_rhyme_tagger import UniTagger


class SchemeScorer:
    def load(self, args):
        self.separator = '\t'
        self.gold = []
        self.out = []
        self.lyrics = []
        with open(args.gold_file, 'r') as gold_input:
            for line in gold_input:
                if self.separator not in line:
                    continue
                line_items = line.split(self.separator)
                self.lyrics.append(line_items[1].strip())
                self.gold.append(line_items[0])
        with open(args.out_file, 'r') as out_input:
            for line in out_input:
                if self.separator not in line:
                    continue
                line_items = line.split(self.separator)
                self.out.append(line_items[0])
                if self.lyrics[len(self.out) - 1] != line_items[1].strip() and len(self.out) <= len(self.lyrics):
                    print(f"Lyrics on line {len(self.out) - 1} don't match. Gold: {self.lyrics[len(self.out) - 1]}. Output: {line_items[1]}")
        if len(self.out) != len(self.gold) or len(self.out) != len(self.lyrics):
            print("ERROR: Scheme lengths don't match")
            print(f"LENGTH GOLDEN: {len(self.gold)}")
            print(f"LENGTH OUTPUT: {len(self.out)}")
            print(f"LENGTH LYRICS: {len(self.lyrics)}")
            return False
        return True

    def compare_direct(self, scheme_gold, scheme_out, lyrics, dist=None, verbose=False):
        self.gold = scheme_gold
        self.out = scheme_out
        self.lyrics = lyrics
        return self.score(dist=dist, verbose=verbose)

    def compare(self, args):
        loadedOK = self.load(args)
        if not loadedOK:
            return
        self.score(dist=args.dist_max)

    @staticmethod
    # Convert scheme to format where each line holds index to the last line it rhymes with.
    def convert_to_last_index_scheme(scheme):
        li_scheme = [-1]*len(scheme)
        for i in range(1, len(scheme)):
            for fellow in range(i-1, -1, -1):
                if scheme[i] == scheme[fellow] and scheme[i] != UniTagger.non_rhyme:
                    li_scheme[i] = fellow
                    break
        return li_scheme

    @staticmethod
    # Assign scheme letters alphabetically. Each non-rhyming line will receive a unique letter.
    def convert_nonrhymes_to_unique(scheme):
        if not scheme:
            return []
        letter_gen = next_letter_generator()
        new_scheme = [next(letter_gen)]
        for i in range(1, len(scheme)):
            letter = ''
            if scheme[i] and scheme[i] != UniTagger.non_rhyme:
                for l in range(i):
                    if scheme[i] == scheme[l]:
                        letter = new_scheme[l]
                        break
            if letter == '':
                letter = next(letter_gen)
            new_scheme.append(letter)
        return new_scheme

    @staticmethod
    # Assign scheme letters alphabetically. Each non-rhyming line will receive the default character.
    def convert_nonrhymes_to_default(scheme, default=UniTagger.non_rhyme):
        if not scheme:
            return []
        new_scheme = [default]
        letter_gen = next_letter_generator()
        for i in range(1, len(scheme)):
            new_scheme.append(default)
            for j in range(i):
                if scheme[i] == scheme[j]:
                    if new_scheme[j] == default:
                        new_scheme[j] = next(letter_gen)
                    new_scheme[i] = new_scheme[j]
                    break
        return new_scheme

    def forbid_distant_rhymes(self, scheme, dist):
        letter_gen = next_letter_generator()
        for i in range(dist, len(scheme)):
            n = i-1
            while n >= 0:
                if scheme[i] == scheme[n]:
                    # Found a match within given distance - rule not broken.
                    if n >= i-dist:
                        break
                    # Found a match too far, change all following matching letters.
                    else:
                        letter = next(letter_gen)
                        while letter in scheme:
                            letter = next(letter_gen)
                        for j in range(i+1, len(scheme)):
                            if scheme[j] == scheme[i]:
                                scheme[j] = letter
                        scheme[i] = letter
                n -= 1

    def score(self, dist=None, verbose=True):
        if dist:
            self.gold = self.forbid_distant_rhymes(self.gold, dist)
            self.out = self.forbid_distant_rhymes(self.out, dist)
        # ARI score
        self.gold = self.convert_nonrhymes_to_unique(self.gold)
        self.out = self.convert_nonrhymes_to_unique(self.out)
        ari_score = sklearn.metrics.adjusted_rand_score(self.gold, self.out)
        # Last index score
        li_gold = self.convert_to_last_index_scheme(self.gold)
        li_out = self.convert_to_last_index_scheme(self.out)
        li_score = sum([1 if li_gold[i] == li_out[i] else 0 for i in range(len(li_gold))])/len(li_gold)
        if verbose:
            print('GOLD | OUT | LYRICS')
            print('-'*20)
            for i in range(len(self.lyrics)):
                print(f"{self.gold[i]:<2} {self.out[i]:<2} {self.lyrics[i]}")
            print(f"ARI SCORE: {ari_score}")
            print(f"Last index score: {li_score}")
        return ari_score, li_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', required=True)
    parser.add_argument('--out_file', required=True)
    # Maximal rhyme distance allowed - scheme will be adjusted if further.
    parser.add_argument('--dist_max',  default=False)
    # Different stanzas can't share a scheme letter.
    parser.add_argument('--stanza_unique',  default=False, action='store_true')
    args = parser.parse_args()
    ss = SchemeScorer()
    ss.compare(args)
