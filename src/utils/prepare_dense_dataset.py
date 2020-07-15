"""Prepare Dense Dataset.

Prepares a subset of the main dataset such that MusicDataset can parse it.
Useful to generate the dense dataset without re-rendering the entire dataset
each time.
"""

from pathlib import Path
import shutil
from argparse import ArgumentParser
from utils.ask import ask
import csv
from tqdm import tqdm
from os import mkdir


def parse_args():
    p = ArgumentParser(description='Creates just a images_svg dataset')
    p.add_argument('SRC', type=str,
                   help='source dataset directory')
    p.add_argument('DEST', type=str,
                   help='destination dataset directory')

    return p.parse_args()


def main(src, dest):
    src = Path(src)
    dest = Path(dest)

    train_names_fp = src / 'train_names.csv'
    test_names_fp = src / 'test_names.csv'

    # First check if src_dir is has the csv files necessary
    if not (train_names_fp.exists() and test_names_fp.exists()):
        raise FileNotFoundError("Names csv files don't exist")

    file_names = []
    with open(train_names_fp) as train_names_csv:
        reader = csv.reader(train_names_csv)
        for line in reader:
            file_names.append(line[1])

    with open(test_names_fp) as test_names_csv:
        reader = csv.reader(test_names_csv)
        for line in reader:
            file_names.append(line[1])

    # Create destination directories
    svg_dir = dest / 'images_svg'
    if dest.exists():
        if svg_dir.exists():
            if not ask('images_svg dir exists. Continue?'):
                return
        else:
            if not ask('dest exists but is empty. Continue?'):
                return
            else:
                mkdir(str(svg_dir))
    else:
        mkdir(dest)
        mkdir(svg_dir)

    # Copy mapping_MuNG file
    shutil.copy2(src / 'mappings_MuNG.csv',
                 dest)
    # Copy name files
    shutil.copy2(src / 'train_names.csv',
                 dest)
    shutil.copy2(src / 'test_names.csv',
                 dest)

    for file in tqdm(file_names, unit='files'):
        cp_from = src / 'images_svg' / file
        cp_from = cp_from.with_suffix('.svg')
        if not cp_from.exists():
            continue

        shutil.copy2(cp_from, dest / 'images_svg')

if __name__ == '__main__':
    args = parse_args()
    main(args.SRC, args.DEST)
