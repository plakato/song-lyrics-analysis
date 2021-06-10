import argparse
import json

import sklearn
from rhymetagger import RhymeTagger

from evaluation.rhyme_detector_v3 import RhymeDetector


class SchemeComparer:
    def __init__(self, args):
        # Default character used for non-rhyming lines' as scheme letter.
        self.non_rhyme = "-"
        self.initialize_tagger()
        self.initialize_detector(args)
        with open(args.file) as data_input:
            self.data = json.load(data_input)

    def initialize_tagger(self):
        self.rt = RhymeTagger()
        self.rt.new_model('en', ngram=4, prob_ipa_min=0.95)
        print('Initialized Rhyme Tagger.')

    def initialize_detector(self, args):
        self.detector_v3 = RhymeDetector(False, args.detector_matrix_file)
        print('Initialized Rhyme Detector v3.')

    def convert_none_to_default_char(self, scheme):
        new_scheme = []
        for l in scheme:
            if l:
                new_scheme.append(l)
            else:
                new_scheme.append(self.non_rhyme)
        return new_scheme

    def compare(self):
        print('Comparing...')
        for song in self.data:
            print(f"SONG: {song['title']}")
            tagger_scheme = self.rt.tag(song['lyrics'], output_format=3)
            tagger_scheme = self.convert_none_to_default_char(tagger_scheme)
            stats_v3 = self.detector_v3.analyze_lyrics(song['lyrics'])
            self.compare_and_print(tagger_scheme, stats_v3['scheme'], song['lyrics'])

    @staticmethod
    def compare_and_print(scheme_gold, scheme_out, lyrics):
        if len(scheme_out) != len(scheme_gold) and len(scheme_out) != len(lyrics):
            print("ERROR: Scheme lengths don't match")
            print(f"LENGTH GOLDEN: {len(scheme_gold)}")
            print(f"LENGTH OUTPUT: {len(scheme_out)}")
            print(f"LENGTH LYRICS: {len(lyrics)}")
            return
        for i in range(len(lyrics)):
            print(f"{scheme_gold[i]:<2} {scheme_out[i]:<2} {lyrics[i]}")
        score = sklearn.metrics.adjusted_rand_score(scheme_gold, scheme_out)
        print(f"SCORE: {score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--detector_matrix_file')
    args = parser.parse_args(['--file', 'data/test_lyrics0.001.json',
                              '--detector_matrix_file', 'data/cooc_iter2.json'])
    sc = SchemeComparer(args)
    sc.compare()