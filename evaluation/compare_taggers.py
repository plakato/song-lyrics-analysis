#!/usr/bin/env python
import argparse
import os
from subprocess import Popen, PIPE

from scheme_scorer import SchemeScorer
from universal_rhyme_tagger import UniTagger


def load_lyrics_annotated(directory):
    songs = []
    schemes = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            filename = os.path.join(subdir, file)
            song = []
            scheme = []
            with open(filename, 'r') as song_file:
                for line in song_file:
                    if line.strip():
                        items = line.strip().split(';')
                        scheme.append(items[0])
                        song.append(items[1])
            songs.append(song)
            schemes.append(scheme)
    return schemes, songs


def load_reddy(filename):
    with open(filename, 'r') as gold_input:
        stanzas = []
        schemes = []
        stanza = []
        scheme = []
        for line in gold_input:
            if line.strip():
                line_items = line.split('\t')
                scheme.append(line_items[0])
                stanza.append(line_items[1].strip())
            elif stanza:
                schemes.append(scheme)
                stanzas.append(stanza)
                scheme = []
                stanza = []
    return schemes, stanzas


def compare(gold_schemes, out_schemes, stanzas, verbose):
    ss = SchemeScorer()
    summed_ari = 0
    summed_li = 0
    weighted_summed_ari = 0
    weighted_summed_li = 0
    for i in range(len(stanzas)):
        ari_score, li_score = ss.compare_direct(gold_schemes[i], out_schemes[i], stanzas[i], verbose=verbose)
        summed_ari += ari_score
        summed_li += li_score
        weighted_summed_ari += ari_score*len(stanzas[i])
        weighted_summed_li += li_score*len(stanzas[i])
    weighted_ari = weighted_summed_ari/sum(map(len, stanzas))
    weighted_li = weighted_summed_li/sum(map(len, stanzas))
    print(f'Average ARI score is {summed_ari/len(stanzas)}.')
    print(f'Weighted average ARI score is {weighted_ari}.')
    print(f'Average last index score is {summed_li/len(stanzas)}.')
    print(f'Weighted average last index score is {weighted_li}.')


def get_schemes(source, stanzas, verbose=False):
    schemes = []
    for stanza in stanzas:
        if source == 'tagger':
            scheme = UniTagger.tagger_tag(stanza)
        elif source == 'v3':
            scheme = UniTagger.detector_v3_tag('data/cooc_iter4.json', stanza, verbose=verbose)
        elif source == 'v3_1st_iter':
            scheme = UniTagger.detector_v3_tag('data/cooc_iter0.json', stanza, verbose=verbose)
        elif source == 'v3_experiment':
            scheme = UniTagger.detector_v3_tag('data/cooc_statistical_0.01.json', stanza, verbose=verbose)
        schemes.append(scheme)
    return schemes


def run_on_reddy_data(args):
    for filename in os.listdir('data_reddy/parsed/'):
        if not filename.endswith('.dev'):
            continue
        print('-' * 20)
        print(f'Analyzing file: {filename}')
        schemes, stanzas = load_reddy('data_reddy/parsed/' + filename)
        # Get gold and output data.
        if args.gold == 'reddy':
            gold = schemes
        else:
            gold = get_schemes(args.gold, stanzas)
        if args.out == 'reddy':
            out = schemes
        else:
            out = get_schemes(args.out, stanzas)
        compare(gold, out, stanzas, args.verbose)


if __name__ == '__main__':
    scheme_sources = ['reddy', 'tagger', 'v3', 'v3_experiment', 'v3_1st_iter', 'lyrics_annotated']
    data_sources = ['reddy','lyrics_annotated']
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--data', required=True, choices=data_sources)
    parser.add_argument('--gold', required=True, choices=scheme_sources)
    parser.add_argument('--out', required=True, choices=scheme_sources)
    # args = parser.parse_args()
    args = parser.parse_args(['--data', 'lyrics_annotated',
                              '--gold', 'lyrics_annotated',
                              '--out', 'v3',
                              '--verbose'
                              ])
    print(f"Using data {args.data} to evaluate schemes by {args.out} in comparison to gold schemes by {args.gold}.")
    if args.data == 'reddy':
        run_on_reddy_data(args)
    elif args.data == 'lyrics_annotated':
        schemes, stanzas = load_lyrics_annotated('data/scheme_annotated/')
        if args.gold == 'lyrics_annotated':
            gold = schemes
        else:
            gold = get_schemes(args.out, stanzas)
        if args.out == 'lyrics_annotated':
            out = schemes
        else:
            out = get_schemes(args.out, stanzas)
        compare(gold, out, stanzas, args.verbose)
