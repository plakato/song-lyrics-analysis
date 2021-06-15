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
        out_scheme = UniTagger.detector_v3_tag('data/cooc_iter4.json', stanzas[i], verbose=verbose)
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


def compare_goldreddy_vs_rt(schemes, stanzas, verbose):
    ss = SchemeScorer()
    summed_ari = 0
    summed_li = 0
    for i in range(len(stanzas)):
        tagger_scheme = UniTagger.tagger_tag(stanzas[i])
        ari_score, li_score = ss.compare_direct(schemes[i], tagger_scheme, stanzas[i], verbose=verbose)
        summed_ari += ari_score
        summed_li += li_score
    print(f'Average ARI score for RhymeTagger is {summed_ari/len(stanzas)}.')
    print(f'Average last index score for RhymeTagger is {summed_li/len(stanzas)}.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', required=True, action='store_true')
    parser.add_argument('--dir')
    args = parser.parse_args(['--dir', 'data_reddy/parsed/', '--verbose'])
    for filename in os.listdir(args.dir):
        if filename.endswith('.dev'):
            print(f'Analyzing file: {filename}')
            schemes, stanzas = load_reddy(args.dir+filename)
            compare_goldreddy_vs_rt(schemes, stanzas, args.verbose)
            compare_goldreddy_vs_v3(schemes, stanzas, args.verbose)
