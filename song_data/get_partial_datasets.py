import json

filename = 'ENlyrics_cleaned.json'
subsets = [100, 1000, 10*1000, 100*1000]


def generate(size, data):
    with open('data/' + str(size) + filename, 'w+') as output:
        output.write('[\n')
        i = 0
        for song in data:
            if i == size:
                break
            if song['lang'] == 'ENGLISH':
                if i != 0:
                    output.write(',\n')
                json.dump(song, output)
                i += 1
        output.write('\n]')


if __name__ == '__main__':
    with open('data/' + filename) as input:
        data = json.load(input)
        for size in subsets:
            generate(size, data)
