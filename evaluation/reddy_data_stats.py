import argparse
import os


def stats(args):
    total_songs = 0
    total_lines = 0
    total_authors = 0
    for filename in os.listdir(args.dir):
        if not filename.endswith(".txt"):
            continue
        total_authors += 1
        with open(args.dir+filename) as annotated:
            # Analyze all poems from one author.
            for line in annotated:
                line = line.strip()
                if line.startswith('TITLE'):
                    total_songs += 1
                    continue
                elif line == '' or line.startswith('RHYME') or line.startswith('AUTHOR'):
                    continue
                total_lines += 1
    return {'total_songs': total_songs,
            'total_lines': total_lines,
            'avg_lines': total_lines/total_songs,
            'total_authors': total_authors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='data_reddy/english_raw/', help="Directory with reddy english raw data.")
    args = parser.parse_args()
    print(stats(args))
