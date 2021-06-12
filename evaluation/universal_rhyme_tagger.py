import argparse
import sys

from rhymetagger import RhymeTagger

from rhyme_detector_v3 import RhymeDetector


class UniTagger:
    def __init__(self, args):
        # Default character used for non-rhyming lines' as scheme letter.
        self.non_rhyme = "-"
        self.lyrics = self.load_input()
        if args.rhyme_tagger:
            self.tagger_tag()
        elif args.rhyme_detector_v3:
            self.detector_v3_tag(args.rhyme_detector_v3)

    def load_input(self):
        lyrics = []
        for line in sys.stdin:
            lyrics.append(line.strip())
        return lyrics

    def tagger_tag(self):
        self.rt = RhymeTagger()
        self.rt.new_model('en', ngram=4, prob_ipa_min=0.95)
        tagger_scheme = self.rt.tag(self.lyrics, output_format=3)
        scheme = self.convert_none_to_default_char(tagger_scheme)
        # Print the result.
        for i in range(len(scheme)):
            print(f"{scheme[i]}\t{self.lyrics[i]}")
        return tagger_scheme

    def detector_v3_tag(self, file):
        detector_v3 = RhymeDetector(False, file)
        stats_v3 = detector_v3.analyze_lyrics(self.lyrics)
        for i in range(len(self.lyrics)):
            print(f"{stats_v3['scheme'][i]}\t{self.lyrics[i]}")

    def convert_none_to_default_char(self, scheme):
        new_scheme = []
        for l in scheme:
            if l:
                new_scheme.append(l)
            else:
                new_scheme.append(self.non_rhyme)
        return new_scheme


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rhyme_tagger', default=False, action='store_true')
    parser.add_argument('--rhyme_tagger_pretrained', default=None, help="File where the pretrained model is stored.")
    parser.add_argument('--rhyme_detector_v3', default=None, help="File with the stored model (co-occurences matrix).")
    args = parser.parse_args()
    # args = parser.parse_args(['--rhyme_detector_v3', 'data/cooc_iter1.json'])
    ut = UniTagger(args)

