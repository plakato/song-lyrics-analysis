import os
from subprocess import Popen, PIPE

from scheme_scorer import SchemeScorer
from universal_rhyme_tagger import UniTagger


def compare_goldreddy_vs_v3(schemes, stanzas):
    ss = SchemeScorer()
    summed = 0
    for i in range(len(stanzas)):
        out_scheme = UniTagger.detector_v3_tag('data/cooc_iter4.json', stanzas[i], verbose=False)
        score = ss.compare_direct(schemes[i], out_scheme, stanzas[i], verbose=False)
        summed += score
    print(f'Average score for v3 is {summed / len(stanzas)}.')


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


def compare_goldreddy_vs_rt(schemes, stanzas):
    ss = SchemeScorer()
    summed = 0
    for i in range(len(stanzas)):
        tagger_scheme = UniTagger.tagger_tag(stanzas[i])
        score = ss.compare_direct(schemes[i], tagger_scheme, stanzas[i], verbose=False)
        summed += score
    print(f'Average score for RhymeTagger is {summed/len(stanzas)}.')


if __name__=='__main__':
    dir = 'data_reddy/parsed/'
    for filename in os.listdir(dir):
        if filename.endswith('.dev'):
            print(f'Analyzing file: {filename}')
            schemes, stanzas = load_reddy(dir+filename)
            compare_goldreddy_vs_rt(schemes, stanzas)
            compare_goldreddy_vs_v3(schemes, stanzas)
