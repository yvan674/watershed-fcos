"""SVG Checker

Checks if every .ly file has been turned into its SVG version

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    March 04, 2020
"""
import os
import os.path as osp
from argparse import ArgumentParser
import shutil
from tqdm import tqdm


def parse_arguments():
    parser = ArgumentParser(description="checks if every .ly has been turned "
                                        "into its SVG version")

    parser.add_argument('IN', type=str,
                        help='path to the ly_database_comm directory')
    parser.add_argument('OUT', type=str,
                        help='path to the svg_database directory')

    return parser.parse_args()


def check(ly_dir, svg_dir, bad_file):
    svg_files = os.listdir(svg_dir)
    svg_set = set()
    for file in svg_files:
        name = str(file.split('-aug-')[0])
        if 'lg' in name:
            svg_set.add(name)

    ly_files = os.listdir(ly_dir)
    bad_files = []
    count_bad = 0
    for file in ly_files:
        if not "aug" in file and 'lg' in file:
            name = osp.splitext(file)[0]
            if name not in svg_set:
                bad_files.append(name + '\n')
                count_bad += 1

    print(f'Number of files not processed: {count_bad}')

    with open(bad_file, mode='w') as bf:
        bf.writelines(bad_files)


def copy_over(ly_dir, new_dir, bad_file):
    b = '\033[1m'
    n = '\033[0m'
    print(f'copying from {b}{ly_dir}{n} to {b}{new_dir}{n} based on '
          f'{b}{bad_file}{n}')
    with open(bad_file, mode='r') as bf:
        bad_files = [line.rstrip('\n') for line in bf]

    # Make sure the new dir exists
    if not osp.exists(new_dir):
        os.mkdir(new_dir)

    for f in tqdm(bad_files):
        src = osp.join(ly_dir, f + '.ly')
        shutil.copy2(src, new_dir)


if __name__ == '__main__':
    args = parse_arguments()

    bad_file = osp.join(osp.split(osp.split(args.IN)[0])[0], 'bad_file.txt')
    check(args.IN, args.OUT, bad_file)

    new_ly = osp.join(osp.split(osp.split(args.IN)[0])[0], 'ly_new')

    copy_over(args.IN, new_ly, bad_file)
