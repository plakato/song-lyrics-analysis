import argparse
import os
from subprocess import Popen, PIPE

from scheme_scorer import SchemeScorer
from universal_rhyme_tagger import UniTagger


def compare_goldreddy_vs_v3(schemes, stanzas, verbose):
    ss = SchemeScorer()
    summed_ari = 0
    summed_li = 0
    for i in range(len(stanzas)):
        ari_score, li_score = ss.compare_direct(schemes[i], out_scheme, stanzas[i], verbose=verbose)
        summed_ari += ari_score
        summed_li += li_score
    print(f'Average ARI score for v3 is {summed_ari / len(stanzas)}.')
    print(f'Average last index score for v3 is {summed_li / len(stanzas)}.')


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
    for i in range(len(stanzas)):
        ari_score, li_score = ss.compare_direct(gold_schemes[i], out_schemes[i], stanzas[i], verbose=verbose)
        summed_ari += ari_score
        summed_li += li_score
    print(f'Average ARI score is {summed_ari/len(stanzas)}.')
    print(f'Average last index score is {summed_li/len(stanzas)}.')


def get_schemes(source, stanzas, verbose=False):
    schemes = []
    for stanza in stanzas:
        if source == 'tagger':
            scheme = UniTagger.tagger_tag(stanza)
        elif source == 'v3':
            scheme = UniTagger.detector_v3_tag('data/cooc_iter4.json', stanza, verbose=verbose)
        schemes.append(scheme)
    return schemes


if __name__ == '__main__':
    scheme_sources = ['reddy', 'tagger', 'v3']
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--data', required=True, choices=['reddy'])
    parser.add_argument('--gold', required=True, choices=scheme_sources)
    parser.add_argument('--out', required=True, choices=scheme_sources)
    args = parser.parse_args(['--data', 'reddy',
                              '--gold', 'reddy',
                              '--out', 'tagger'])
    print(f"Using data {args.data} to evaluate schemes by {args.out} in comparison to gold schemes by {args.gold}.")
    if args.data == 'reddy':
        for filename in os.listdir('data_reddy/parsed/'):
            if not filename.endswith('.dev'):
                continue
            print('-'* 20)
            print(f'Analyzing file: {filename}')
            schemes, stanzas = load_reddy('data_reddy/parsed/' + filename)
            # Get gold and output data.
            if args.gold == 'reddy':
                gold = schemes
            else:
                gold = get_schemes(args.gold, stanzas, args.verbose)
            if args.out == 'reddy':
                out = schemes
            else:
                out = get_schemes(args.out, stanzas, args.verbose)
            compare(gold, out, stanzas, args.verbose)
