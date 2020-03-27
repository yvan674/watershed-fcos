"""Remove unnecessary images.

When the dataset files are generated, it only generates an annotation file and
do nothing to the original images. This script removes files that aren't in
either the training or validation annotation files.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    March 27, 2020
"""
from os import listdir, remove
from os.path import join, splitext
import json
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser('removes unnecessary images')
    parser.add_argument('DIR', type=str)
    parser.add_argument('TRAIN', type=str)
    parser.add_argument('VAL', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print('Loading annotations...')
    train_json = json.load(join(args.DIR, args.TRAIN))
    val_json = json.load(join(args.DIR, args.VAL))

    files_to_remove = []

    train_files = [splitext(img['filename'])[0] for img in train_json['images']]
    val_files = [splitext(img['filename'])[0] for img in val_json['images']]

    good_files = set(train_files) | set(val_files)

    print('Scanning directories...')
    img_files = listdir(join(args.DIR, 'images_png'))
    seg_files = listdir(join(args.DIR, 'pix_annotations_png'))

    for img in img_files:
        if splitext(img)[0] not in good_files:
            files_to_remove.append(join(args.DIR, 'images_png', img))

    for seg in seg_files:
        if seg.strip('_seg') not in good_files:
            files_to_remove.append(join(args.DIR, 'pix_annotations_png', seg))

    print(f'Removing {len(files_to_remove)} files...')

    for img in tqdm(files_to_remove):
        remove(img)
