"""Splitter.

Splits the DeepScores dataset with a 80:20 training:validation ratio.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 25, 2019
"""
import csv
import argparse
from os import listdir
import random
from os.path import join, splitext


def parse_arguments():
    desc = "splits the DeepScores dataset with an 80:20 training:validation " \
           "ratio."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('DIR', type=str,
                        help='path to directory containing the dataset')

    return parser.parse_args()


def random_split(file_names: list) -> tuple:
    """Randomly splits file_names with an 80:20 ratio."""
    frac = 0.2
    num_files = len(file_names)

    # Get indices of items to remove, i.e. in test set.
    inds = set(random.sample(list(range(num_files)), int(frac * num_files)))

    test_set = []
    train_set = []
    for i, n in enumerate(file_names):
        if i in inds:
            test_set.append(splitext(n)[0])
        else:
            train_set.append(splitext(n)[0])

    return train_set, test_set


def write_csv(file_names: list, csv_file_path: str):
    """Writes the file names to a CSV file."""
    with open(csv_file_path, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(enumerate(file_names))


if __name__ == '__main__':
    args = parse_arguments()

    # First reads all file names
    print("Reading file names...")
    file_names = listdir(join(args.DIR, 'images_png/'))

    print('Splitting into training and test sets...')
    train_set, test_set = random_split(file_names)

    print('Writing training files to disk...')
    write_csv(train_set, join(args.DIR, 'train_names.csv'))

    print('Writing test files to disk...')
    write_csv(test_set, join(args.DIR, 'test_names.csv'))

    print('Split completed!')
