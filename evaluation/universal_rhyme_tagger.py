import argparse
import json
import os
import sys

from rhymetagger import RhymeTagger

from constants import NO_OF_PRECEDING_LINES
from rhyme_detector_v3 import RhymeDetector


class UniTagger:
    non_rhyme = "-"

    def tag(self, args):
        # Default character used for non-rhyming lines' as scheme letter.
        self.lyrics = self.load_input()
        if args.rhyme_tagger:
            self.tagger_tag()
        elif args.rhyme_detector_v3:
            self.detector_v3_tag(args.rhyme_detector_v3, self.lyrics)
        elif args.rhyme_tagger_pretrained:
            self.pretrain_tagger(args.rhyme_tagger_pretrained)
            self.tagger_tag()

    def pretrain_tagger(self, pretrained_file):
        train_file = 'data/train_lyrics0.33.json'
        self.rt = RhymeTagger()
        # Pretrain model if doesn't exist yet.
        if os.path.exists(pretrained_file):
            self.rt.load_model(pretrained_file, verbose=False)
            return self.rt
        print(f"Pretrained model doesn't exist. Pretraining from file {train_file}")
        with open(train_file, 'r') as train_input:
            train_data = json.load(train_input)
        self.rt.new_model(lang='en', window=NO_OF_PRECEDING_LINES, verbose=False)
        for song in train_data:
            self.rt.add_to_model(song['lyrics'])
        self.rt.train_model()
        self.rt.save_model(pretrained_file)
        print(f"Saving pretrained model to {pretrained_file}.")
        return self.rt

    def load_input(self):
        lyrics = []
        for line in sys.stdin:
            lyrics.append(line.strip())
        return lyrics

    def tagger_tag(self, verbose=False, lyrics=None):
        if lyrics:
            self.lyrics = lyrics
        if not hasattr(self, 'rt'):
            self.rt = RhymeTagger()
            self.rt.new_model('en', ngram=4, prob_ipa_min=0.95, window=NO_OF_PRECEDING_LINES, verbose=False)
        tagger_scheme = self.rt.tag(self.lyrics, output_format=3)
        scheme = self.convert_none_to_default_char(tagger_scheme)
        if verbose:
            # Print the result.
            for i in range(len(scheme)):
                print(f"{scheme[i]}\t{self.lyrics[i]}")
        return tagger_scheme


    @staticmethod
    def detector_v3_tag(file, lyrics=None, perfect=False, verbose=False):
        detector_v3 = RhymeDetector(perfect_only=perfect, matrix_path=file, verbose=False)
        stats_v3 = detector_v3.analyze_lyrics(lyrics)
        if verbose:
            for i in range(len(lyrics)):
                print(f"{stats_v3['scheme'][i]}\t{lyrics[i]}")
        return stats_v3['scheme']

    @staticmethod
    def convert_none_to_default_char(scheme):
        new_scheme = []
        for l in scheme:
            if l:
                new_scheme.append(l)
            else:
                new_scheme.append(UniTagger.non_rhyme)
        return new_scheme


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rhyme_tagger', default=False, action='store_true')
    parser.add_argument('--rhyme_tagger_pretrained', default=None, help="File where the pretrained model is stored.")
    parser.add_argument('--rhyme_detector_v3', default=None, help="File with the stored model (co-occurences matrix).")
    # args = parser.parse_args()
    args = parser.parse_args(['--rhyme_tagger_pretrained', 'tagger_pretrained_on_lyrics-train0.3.model.json'])
    ut = UniTagger()
    ut.tag(args)

