import argparse
import os
from os.path import isfile, join
import numpy as np
from repetition_matrix import generate_endrhyme_matrix


def test_individual(path, shuffled_path):
    success = 0
    fail = 0
    count = 0
    for item in os.listdir(path):
        song = join(path, item)
        if isfile(song) and item.endswith('.csv'):
            count += 1
            is_fake = shuffled_path in song
            matrix = generate_endrhyme_matrix(path, item)
            judged_true = is_song()
            if (judged_true and not is_fake) or (is_fake and not judged_true):
                success += 1
                print('Successfully classified',  'fake' if is_fake else
                'original', 'in', item)
            else:
                fail += 1
                print('Failed classifying in', item)
    print('Correctly recognized {0}/{1}, which is {2}%'.format(success,
                                                                  count,
                                                              100*success/count))


def test_pairs(path, shuffled_path):
    success = 0
    fail = 0
    count = 0
    for item in os.listdir(path):
        original = join(path, item)
        shuffled = join(path, shuffled_path + item[1:])
        if isfile(original) and isfile(shuffled) and item.endswith('.csv'):
            count += 1
            original = generate_endrhyme_matrix(path, item)
            shuffled = generate_endrhyme_matrix(path, shuffled_path + item[1:])
            if choose_original(original, shuffled):
                success += 1
                print('Successfully recognized original in', item)
            else:
                fail += 1
                print('Failed recognizing original in', item)
    print('Correctly recognized {0}/{1} pairs, which is {2}%'.format(success,
                                                                  count,
                                                              100*success/count))


def choose_original(original, shuffled):
    original_islands = count_lonely_black_pixels(original)
    shuffled_islands = count_lonely_black_pixels(shuffled)
    original_white_square = get_size_of_the_largest_white_square(original)
    shuffled_white_square = get_size_of_the_largest_white_square(shuffled)
    original_furthest_black = max(get_distances_to_closest_blacks(original))
    shuffled_furthest_black = max(get_distances_to_closest_blacks(shuffled))
    if original_islands != shuffled_islands:
        result = original_islands < shuffled_islands
    else:
        result = original_white_square < shuffled_white_square
    return result


# Count islands on half of matrix (the other half is symmetrical).
def count_lonely_black_pixels(data):
    lonely_count = 0
    for x in range(len(data)):
        for y in range(x):
            if data[x, y]:
                # Look at the surrounding pixels.
                white = True
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        # Is not out of grid.
                        if x+i > -1 and y+j > -1 and x+i < len(data) and \
                                y+j<len(data[x]):
                            if data[x+i, y+j] and (i != 0 or j != 0):
                                white = False
                if white:
                    lonely_count += 1
    return lonely_count


def get_size_of_the_largest_white_square(data):
    # Create sum matrix.
    sum_matrix = np.zeros(data.shape)
    # Copy first row and first column.
    for i in range(len(data)):
        if data[0, i]:
            sum_matrix[0, i] = 0
        else:
            sum_matrix[0, i] = 1
        if data[i, 0]:
            sum_matrix[i, 0] = 0
        else:
            sum_matrix[i, 0] = 1

    # Helper function to calculate one square in sum matrix.
    def calculate_value(k, l):
        if not data[k, l]:
            sum_matrix[k, l] = min(sum_matrix[k - 1, l - 1],
                                   sum_matrix[k - 1, l],
                                   sum_matrix[k, l - 1]) + 1
        else:
            sum_matrix[k, l] = 0

    # Fill in the rest.
    for i in range(1, len(data)):
        for j in range(i, len(data)):
            calculate_value(i, j)
            calculate_value(j, i)
    return max(map(max, sum_matrix))


# Return list of distances from diagonal to closest black pixel to the right.
def get_distances_to_closest_blacks(matrix):
    res = []
    i = 0
    while i < len(matrix):
        y = i
        while y < len(matrix):
            if y + 1 == len(matrix) or matrix[i][y + 1] == 1:
                res.append(y-i)
                break
            y += 1
        i += 1
    return res


def is_song(matrix):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='sparsar_experiments/rhymes/')
    args = parser.parse_args()
    shuffled_path = '\'shuffled0_'
    test_pairs(args.path, shuffled_path)
    test_individual(args.path, shuffled_path)