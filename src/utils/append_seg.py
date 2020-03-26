"""Append Seg.

Appends "_seg" to all *.png files in a directory.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    March 26, 2020
"""
from os.path import splitext, join
from os import listdir, rename
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(
        description='appends _seg to all *.png files in the given directory'
    )
    parser.add_argument('DIR', type=str)
    return parser.parse_args()


def main(dir_path):
    files = listdir(dir_path)

    for file in tqdm(files):
        file_name = splitext(file)
        if '.png' not in file_name[1]:
            continue
        old_name = join(dir_path, file)
        new_name = join(dir_path, file_name[0] + '_seg' + file_name[1])
        rename(old_name, new_name)


if __name__ == '__main__':
    args = parse_args()
    main(args.DIR)
