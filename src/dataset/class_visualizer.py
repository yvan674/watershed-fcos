"""Class Visualizer.

This script makes it possible to crops out every annotation and puts them in a
folder so it is easy to visualize every instance of every class.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created On:
    18 May 2020.
"""
from PIL import Image
import json
from shutil import rmtree
from os.path import exists, isabs, join
from os import makedirs
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    """Parses command line arguments."""
    p = ArgumentParser(description='crops out all individual instances of each '
                                   'class in the dataset.')

    p.add_argument('DIR', type=str,
                   help='path to the dataset directory')
    p.add_argument('ANN', type=str,
                   help='path to the annotation file. Relative if inside DIR, '
                        'otherwise absolute')
    p.add_argument('OUT', type=str,
                   help='path to where the output files should be generated')

    return p.parse_args()


def main(dataset_fp, ann_fp, out_fp):
    """Main function.

    Note:
        Merge functionality exists so we can add both the train and validation
        annotations to the same dir.

    :param str dataset_fp: Path to the dataset directory.
    :param str ann_fp: Path to the annotation file. Relative if inside
        dataset_fp, otherwise absolute.
    :param str out_fp: Path to where the output should be.
    """
    mkdirtree = True
    # First check if out_fp exists
    if exists(out_fp):
        # Then ask what to do with it
        user_response = ""
        pos_response = ('q', 'Q', 'o', 'O', 'm', 'M')
        prompt = "OUT dir already exists. [Q]uit, [O]verwrite, or [M]erge?"
        while user_response not in pos_response:
            user_response = input(prompt)

        if user_response in ('q', 'Q'):
            return
        elif user_response in ('o', 'O'):
            rmtree(out_fp)
        elif user_response in ('m', 'M'):
            mkdirtree = False
        else:
            raise ValueError('User responded with something unexpected.')

    # Load the annotation file first so we know which classes to make a dir for.
    if not isabs(ann_fp):
        ann_fp = join(dataset_fp, ann_fp)
    with open(ann_fp) as ann_file:
        anns = json.load(ann_file)

    ann_sets = anns['annotation_sets']
    categories = {i: {} for i in ann_sets}
    for k, v in anns['categories'].items():
        categories[v['annotation_set']][k] = v['name']

    if mkdirtree:
        # Make the directory tree if necessary.
        for ann_set in ann_sets:
            for _, v in categories[ann_set].items():
                makedirs(join(out_fp, ann_set, v))

    # Now go through each image to reduce the number of file ops needed.
    counter = 0
    for img_ann in tqdm(anns['images'], unit='imgs'):
        img_fp = img_ann['filename']
        img_objects = img_ann['ann_ids']
        img = Image.open(join(dataset_fp, 'images', img_fp))
        for ann_id in img_objects:
            crop_box = anns['annotations'][ann_id]['a_bbox']
            crop_box[2] += crop_box[0]
            crop_box[3] += crop_box[1]
            if crop_box[2] - crop_box[0] < 2:
                crop_box[2] += 2
            if crop_box[3] - crop_box[1] < 2:
                crop_box[3] += 2
            cropped_image = img.crop(crop_box)
            for i, ann_set in enumerate(ann_sets):
                obj_cat_id = anns['annotations'][ann_id]['cat_id'][i]
                obj_cat = categories[ann_set][obj_cat_id]
                cropped_image.save(join(out_fp, ann_set, obj_cat, str(counter))
                                   + '.png')
                counter += 1


if __name__ == '__main__':
    args = parse_args()
    main(args.DIR, args.ANN, args.OUT)
