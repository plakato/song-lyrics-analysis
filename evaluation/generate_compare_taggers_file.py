import argparse
import csv
import re
import subprocess
from datetime import datetime


def get_scores(data, tagger):
    command = ['python', './compare_taggers.py', '--data', data, '--gold', data, '--out', tagger, '--verbose']
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    scores_str = result.split('\n')[-5:-1]
    float_num = re.compile(r'\d\.[\d]+')
    scores = [round(float(re.search(float_num, score).group(0)), 4) for score in scores_str]
    return scores


def main(args):
    tagger_sources = ['tagger', 'tagger_pretrained', 'v3', 'v3_experiment', 'v3_1st_iter', 'v3_perfect']
    data_sources = ['reddy', 'lyrics_annotated_dev']
    header = ['', 'macro ARI', 'micro ARI', 'macro LI', 'micro LI']
    with open(args.out, 'w+') as f:
        writer = csv.writer(f)
        for data in data_sources:
            writer.writerow([f'Data source: {data}'])
            writer.writerow(header)
            for tagger in tagger_sources:
                scores = get_scores(data, tagger)
                writer.writerow([tagger] + scores)
            writer.writerow([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    out_file = f'results/tagger_comparison/scores_{date}.csv'
    parser.add_argument('--out', default=out_file, help="Name of the csv file, where the output will be saved.")
    args = parser.parse_args()
    main(args)

