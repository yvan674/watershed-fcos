"""Smallifier.

Creates a small version of a COCO dataset.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    December 10, 2019
"""
import argparse
from pycocotools.coco import COCO
import random
import json


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='creates a small version of '
                                                 'a COCO dataset')

    parser.add_argument('ANNOTATIONS', type=str,
                        help='path to the annotations file')
    parser.add_argument('N', type=int,
                       help='number of images to pick')
    parser.add_argument('OUTPUT', type=str,
                        help='path to the output annotation file')

    return parser.parse_args()


def pick_images(coco: COCO, n: int) -> list:
    """Picks n images.

    Args:
        coco: The COCO object to pick images from.
        n: Number of images to pick.

    Returns:
        A list of image IDs.
    """
    print('Reading images and picking {}...'.format(n))

    # Get indices of items we want
    inds = set(random.sample(list(range(len(coco.imgs))), n))

    # Get img keys so
    img_ids = list(coco.imgs.keys())

    # Return a list of all coco img ids from the list of inds
    return [img_ids[ind] for ind in inds]


def generate_new_annotations(ann_ids: list, img_ids: list,
                             coco: COCO, output_fp: str):
    """Generates the new annotations file.

    Args:
        ann_ids: The list of annotations IDs to add.
        img_ids: List of image IDs to add.
        coco: The COCO object.
        output_fp: Where to output the new annotation file to.
    """
    print('Loading categories')
    coco_cats = coco.cats
    cats = [coco_cats[cat] for cat in coco_cats.keys()]
    print('Loading new images and annotations...')
    new_annotations = {
        'info': {
            'description': 'A small version of the DeepScores dataset.',
            'url': 'https://tuggeluk.github.io/deepscores/',
            'version': '1.0'
        },

        'licenses': [
            {
                'url': 'https://opensource.org/licenses/MIT',
                'id': 1,
                'name': 'MIT License'
            }
        ],
        'images': coco.loadImgs(img_ids),
        'annotations': coco.loadAnns(ann_ids),
        'categories': cats
    }

    print('Writing new annotation file to disk...')
    with open(output_fp, 'w+') as output_file:
        json.dump(new_annotations, output_file)


if __name__ == '__main__':
    args = parse_arguments()

    coco = COCO(args.ANNOTATIONS)

    images = pick_images(coco, args.N)

    print('Getting annotation IDs for the chosen images...')
    ann_ids = coco.getAnnIds(images)

    generate_new_annotations(ann_ids, images, coco, args.OUTPUT)
    print('Done!')


