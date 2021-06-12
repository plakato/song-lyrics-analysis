import argparse

import sklearn.metrics


class SchemeScorer:
    def load(self, args):
        self.separator = '\t'
        self.gold = []
        self.out = []
        self.lyrics = []
        with open(args.gold_file, 'r') as gold_input:
            for line in gold_input:
                if self.separator not in line:
                    continue
                line_items = line.split(self.separator)
                self.lyrics.append(line_items[1].strip())
                self.gold.append(line_items[0])
        with open(args.out_file, 'r') as out_input:
            for line in out_input:
                if self.separator not in line:
                    continue
                line_items = line.split(self.separator)
                self.out.append(line_items[0])
                if self.lyrics[len(self.out) - 1] != line_items[1].strip() and len(self.out) <= len(self.lyrics):
                    print(f"Lyrics on line {len(self.out) - 1} don't match. Gold: {self.lyrics[len(self.out) - 1]}. Output: {line_items[1]}")
        if len(self.out) != len(self.gold) or len(self.out) != len(self.lyrics):
            print("ERROR: Scheme lengths don't match")
            print(f"LENGTH GOLDEN: {len(self.gold)}")
            print(f"LENGTH OUTPUT: {len(self.out)}")
            print(f"LENGTH LYRICS: {len(self.lyrics)}")
            return False
        return True

    def compare(self, args):
        loadedOK = self.load(args)
        if not loadedOK:
            return
        self.print_score()

    def print_score(self):
        print('GOLD | OUT | LYRICS')
        print('-'*20)
        for i in range(len(self.lyrics)):
            print(f"{self.gold[i]:<2} {self.out[i]:<2} {self.lyrics[i]}")
        score = sklearn.metrics.adjusted_rand_score(self.gold, self.out)
        print(f"ARI SCORE: {score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', required=True)
    parser.add_argument('--out_file', required=True)
    # Maximal rhyme distance allowed - scheme will be adjusted if further.
    parser.add_argument('--dist_max',  default=False)
    # Different stanzas can't share a scheme letter.
    parser.add_argument('--stanza_unique',  default=False, action='store_true')
    args = parser.parse_args()
    ss = SchemeScorer()
    ss.compare(args)
