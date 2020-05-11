"""Deepscores Converstion to Oriented Bounding Box Annotations.

Converts the deepscores dataset to the Oriented Bounding Box Annotations.
pretty_time_delta() was taken from a gist by thatalextaylor:
https://gist.github.com/thatalextaylor/7408395

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    March 03, 2020
"""
import os.path as osp
import os
from shutil import rmtree
from conversion import *
import csv
import argparse
from sys import exit
from time import time
from tqdm import tqdm


def pretty_time_delta(seconds):
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)


def parse_argument():
    desc = 'converts the DeepScores dataset to conform to the Oriented ' \
           'BoundingBox dataset schema.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('DIR', type=str,
                        help='path to the directory containing the dataset')
    parser.add_argument('CLASSES', type=str,
                        help='path to the file containing the class names list')
    parser.add_argument('-o', '--oriented', action='store_true',
                        help='output using OBB schema')

    return parser.parse_args()


def ask(question: str = 'Continue?') -> bool:
    """Asks the user to continue or not."""
    response = input(f'{question}[y/n] ')
    while True:
        if response == 'n':
            return False
        elif response == 'y':
            return True
        else:
            response = input('Please type [y] or [n] ')


def do_conversion(dir_path: str, class_names_fp: str, obb: bool) -> tuple:
    """Does the actual conversion.

    Returns:
        A tuple where the first element is the training OBBAnnotations
        and the second element is the validation OBBAnnotations.
    """
    training_set = set()
    val_set = set()

    # Get the training set and validation split as sets
    with open(osp.join(dir_path, 'train_names.csv')) as train_names:
        reader = csv.reader(train_names)
        for line in reader:
            training_set.add(line[1])

    try:
        with open(osp.join(dir_path, 'test_names.csv')) as val_names:
            reader = csv.reader(val_names)
            for line in reader:
                val_set.add(line[1])
    except FileNotFoundError:
        print("WARNING: test_names.csv has not been found. Assuming every file "
              "not in train_names.csv is part of the validation set.")
        val_set = None

    # Make sure the work directory exists
    work_dir = osp.join(dir_path, 'tmp')
    if not osp.exists(work_dir):
        os.mkdir(work_dir)

    print('Processing image information...')
    # Process the image directory
    train_img_path, val_img_path, img_lookup = process_image_dir(
        osp.join(dir_path, 'images_png'), work_dir, training_set, val_set
    )

    # Verification step
    num_train_imgs = -1  # To account for header row
    num_test_imgs = -1
    with open(train_img_path, 'r') as file:
        for _ in file:
            num_train_imgs += 1

    with open(val_img_path, 'r') as file:
        for _ in file:
            num_test_imgs += 1

    print(f'Found {num_train_imgs} training images and {num_test_imgs} '
          'validation images.')

    if not ask():
        if ask('Delete tmp files?'):
            # Deletes all tmp files
            rmtree(osp.join(dir_path, 'tmp'), ignore_errors=True)
        exit()

    # Process the segmentation directory
    print('Checking segmentation directory...')
    seg_dir = osp.join(dir_path, 'pix_annotations_png')
    seg_files = os.listdir(osp.join(dir_path, 'pix_annotations_png'))
    for file in tqdm(seg_files):
        # Make sure that _seg has been added to the file
        ext = osp.splitext(file)
        if '.png' not in ext:
            # We only expect _seg in png file names so we ignore the file if
            # it's not a png file
            continue
        elif '_seg' in file:
            continue
        else:
            os.rename(osp.join(seg_dir, file),
                      osp.join(seg_dir, ext[0] + '_seg' + ext[1]))
    print('Done!')

    # Figure out the classes
    print("Reading categories...")
    if not osp.isabs(class_names_fp):
        class_names_fp = osp.join(dir_path, class_names_fp)
    if obb:
        categories = generate_oriented_categories(class_names_fp)
    else:
        categories, cat_lookup, cat_set = generate_categories(class_names_fp)
    print("Done!")

    # Process the annotations
    print("Generating annotations...")
    if obb:
        annotations = generate_oriented_annotations(
            pix_annotations_dir=osp.join(dir_path, 'pix_annotations_png'),
            xml_annotations_dir=osp.join(dir_path, 'xml_annotations'),
            categories=categories,
            img_lookup=img_lookup,
            train_set=training_set,
            val_set=val_set
        )
    else:
        annotations = generate_annotations(
            pix_annotations_dir=osp.join(dir_path, 'pix_annotations_png'),
            xml_annotations_dir=osp.join(dir_path, 'xml_annotations'),
            category_lookup=cat_lookup,
            img_lookup=img_lookup,
            train_set=training_set,
            val_set=val_set,
            oriented=obb
        )
        print("Done!")
    train_ann, val_ann, train_ann_lookup, val_ann_lookup = annotations

    if obb:
        ds = OBBAnnotations
        train_desc = "Deepscores training set in the OBB format"
        val_desc = "Deepscores validation set in the OBB format"
        img_style = 'obb'
    else:
        ds = CocoLikeAnnotations
        train_desc = "Deepscores training set in the COCO format"
        val_desc = "Deepscores validation set in the COCO format"
        img_style = 'coco'

    # Once that's complete, generate the actual dataset objects.
    train_dataset = ds(train_desc)
    val_dataset = ds(val_desc)

    train_dataset.add_images(image_csv_to_dict(train_img_path, img_style,
                                               train_ann_lookup))
    val_dataset.add_images(image_csv_to_dict(val_img_path, img_style,
                                             val_ann_lookup))

    if obb:
        # TODO add Muscima dataset annotations as well.
        train_dataset.add_categories(categories, 'deepscores')
        val_dataset.add_categories(categories, 'deepscores')
    else:
        train_dataset.add_categories(categories)
        val_dataset.add_categories(categories)

    train_dataset.add_annotations(train_ann)
    val_dataset.add_annotations(val_ann)

    return train_dataset, val_dataset


if __name__ == '__main__':
    arguments = parse_argument()
    start_time = time()

    train, val = do_conversion(arguments.DIR, arguments.CLASSES,
                               arguments.oriented)

    name_prefix = 'deepscores_oriented' if arguments.oriented else 'deepscores'

    print('\nWriting training annotation file to disk...')
    train.output_json(osp.join(arguments.DIR, name_prefix + '_train.json'))

    print('\nWriting validation annotation file to disk...')
    val.output_json(osp.join(arguments.DIR, name_prefix + '_val.json'))

    print('\nConversion completed!')
    print(f"Total time: {pretty_time_delta(time() - start_time)}")
