"""Copy necessary images.

When the dataset files are generated, it only generates an annotation file and
do nothing to the original images. This script copies files that are in
either the training or validation annotation files to a new dir.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    April 9, 2020
"""
from os import listdir, mkdir
from os.path import join, splitext, exists, split
from shutil import copyfile
import json
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser('copies necessary images')
    parser.add_argument('DIR', type=str)
    parser.add_argument('TRAIN', type=str)
    parser.add_argument('VAL', type=str)
    parser.add_argument('OUT')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print('Loading annotations...')
    train_ann_fp = join(args.DIR, args.TRAIN)
    val_ann_fp = join(args.DIR, args.VAL)
    with open(train_ann_fp) as json_file:
        train_json = json.load(json_file)
    with open(val_ann_fp) as json_file:
        val_json = json.load(json_file)

    imgs_to_copy = []
    seg_to_copy = []

    train_files = [splitext(img['filename'])[0]
                   for img in train_json['images']]
    val_files = [splitext(img['filename'])[0]
                 for img in val_json['images']]

    good_files = set(train_files) | set(val_files)

    print('Scanning directories...')
    img_files = listdir(join(args.DIR, 'images_png'))
    seg_files = listdir(join(args.DIR, 'pix_annotations_png'))

    for img in img_files:
        if splitext(img)[0] in good_files:
            imgs_to_copy.append(join(args.DIR, 'images_png', img))

    for seg in seg_files:
        if seg.strip('_seg.png') in good_files:
            seg_to_copy.append(join(args.DIR, 'pix_annotations_png', seg))

    print(f'Copying {len(imgs_to_copy) + len(seg_to_copy)} files...')

    if not exists(args.OUT):
        mkdir(args.OUT)
        mkdir(join(args.OUT, 'images_png'))
        mkdir(join(args.OUT, 'pix_annotations_png'))

    print("Copying images...")
    for img in tqdm(imgs_to_copy):
        copyfile(img, join(args.OUT, 'images_png', split(img)[1]))

    print("Copying segmentation...")
    for seg in tqdm(seg_to_copy):
        copyfile(seg, join(args.OUT, 'pix_annotations_png', split(seg)[1]))

    print("Copying annotations...")
    copyfile(train_ann_fp, join(args.OUT, args.TRAIN))
    copyfile(val_ann_fp, join(args.OUT, args.VAL))
