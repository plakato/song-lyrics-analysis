import argparse
import os

from rhyme_detector_v1 import next_letter_generator


def parse(args):
    for filename in os.listdir(args.dir):
        if not filename.endswith(".txt"):
            continue
        total_lines = 0
        schemes = []
        lyrics = []
        c = 0
        with open(args.dir+filename) as annotated:
            # Analyze all poems from one author.
            stanza = []
            for line in annotated:
                c += 1
                line = line.strip()
                if line.startswith('AUTHOR') or line.startswith('TITLE') or line.startswith('RHYME-POEM'):
                    continue
                elif line.startswith('RHYME'):
                    stanza_scheme = line.strip().split(' ')[1:]
                    schemes.append(stanza_scheme)
                elif line.strip():
                    stanza.append(line.strip())
                    total_lines += 1
                else:
                    if stanza:
                        lyrics.append(stanza)
                        stanza = []
                    else:
                        continue
        # Split to dev/test.
        dev_count = int(args.dev_test_ratio * total_lines)
        save_dev_and_test(args.out_dir, filename, dev_count, schemes, lyrics)


def save_dev_and_test(out_dir, filename, dev_count, schemes, lyrics):
    dev_filename = out_dir + filename[:-4] + '.dev'
    test_filename = out_dir + filename[:-4] + '.test'
    with open(dev_filename, 'w+') as dev_out:
        count = 0
        last_dev_idx = 0
        for i in range(len(lyrics)):
            count += len(lyrics[i])
            letter_gen = next_letter_generator()
            next_letter = True
            for l in range(len(lyrics[i])):
                # Special case - means aa bb cc...
                if schemes[i][-1] == '*':
                    if next_letter:
                        letter = next(letter_gen)
                        next_letter = False
                    else:
                        next_letter = True
                    dev_out.write(f"{letter}\t{lyrics[i][l]}\n")
                else:
                    # Just copy the scheme.
                    dev_out.write(f"{schemes[i][l]}\t{lyrics[i][l]}\n")
            dev_out.write('\n')
            if count > dev_count:
                last_dev_idx = i
                break
    with open(test_filename, 'w+') as test_out:
        for i in range(last_dev_idx+1, len(lyrics)):
            for l in range(len(lyrics[i])):
                # todo
                # Special case - * means continue as shown
                letter_gen = next_letter_generator()
                next_letter = True
                if schemes[i][-1] == '*':
                    if next_letter:
                        letter = next(letter_gen)
                        next_letter = False
                    else:
                        next_letter = True
                    test_out.write(f"{letter}\t{lyrics[i][l]}\n")
                else:
                    # Just copy the scheme.
                    test_out.write(f"{schemes[i][l]}\t{lyrics[i][l]}\n")
            test_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help="Directory with reddy english raw data.")
    parser.add_argument('--out_dir', required=True, help="Directory where parsed data will be saved.")
    parser.add_argument('--dev_test_ratio', type=float)
    args = parser.parse_args(['--dir', 'data_reddy/rhymedata/english_raw/',
                              '--out_dir', 'data_reddy/parsed/',
                              '--dev_test_ratio', '0.3'])
    parse(args)
