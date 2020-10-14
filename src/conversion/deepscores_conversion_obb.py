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
from pathlib import Path
from shutil import rmtree
from conversion import *
import csv
import argparse
from sys import exit
from time import time
from tqdm import tqdm
from utils.ask import ask


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
    parser.add_argument('CLASSES', type=str, nargs='?', default='',
                        help='path to the file containing the class names list')
    parser.add_argument('-d', '--dense', action='store_true',
                        help='dataset should be marked as dense')
    parser.add_argument('-y', action='store_true',
                        help='silently say yes to all input queries.')

    return parser.parse_args()


def do_conversion(dir_path: str, class_names_fp: str, dense: bool, silent: bool):
    """Does the actual conversion."""
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
    if not silent:
        if not ask():
            if ask('Delete tmp files?'):
                # Deletes all tmp files
                rmtree(osp.join(dir_path, 'tmp'), ignore_errors=True)
            exit()

    # Process the segmentation directory
    print('Checking segmentation directory...')
    seg_dir = osp.join(dir_path, 'pix_annotations_png')
    seg_files = os.listdir(seg_dir)
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

    # Process the instance directory
    print('Checking instance directory...')
    inst_dir = osp.join(dir_path, 'instance_png')
    inst_files = os.listdir(inst_dir)
    for file in tqdm(inst_files):
        # Make sure that _seg has been added to the file
        ext = osp.splitext(file)
        if '.png' not in ext:
            # We only expect _seg in png file names so we ignore the file if
            # it's not a png file
            continue
        elif '_inst' in file:
            continue
        else:
            os.rename(osp.join(inst_dir, file),
                      osp.join(inst_dir, ext[0] + '_inst' + ext[1]))
    print('Done!')

    # Figure out the classes
    print("Reading categories...")
    if not osp.isabs(class_names_fp):
        class_names_fp = osp.join(dir_path, class_names_fp)
    categories = generate_oriented_categories(class_names_fp)
    print("Done!")

    # Process the annotations
    print("Generating annotations...")
    annotations = generate_oriented_annotations(
        instance_dir=inst_dir,
        xml_annotations_dir=osp.join(dir_path, 'xml_annotations'),
        categories=categories,
        img_lookup=img_lookup,
        work_dir=work_dir,
        train_set=training_set,
        val_set=val_set
    )
    print("Done!")
    train_ann_fps, val_ann_fps, train_ann_lookup, val_ann_lookup = annotations

    train_desc = "Deepscores training set in the OBB format"
    val_desc = "Deepscores test set in the OBB format"
    img_style = 'obb'

    # Once that's complete, generate the actual dataset objects.
    if dense:
        train_dataset = OBBAnnotations(train_desc, version='2.0',
                                       subset='dense')
        val_dataset = OBBAnnotations(val_desc, version='2.0',
                                     subset='dense')

        train_dataset.add_images(image_csv_to_dict(train_img_path, img_style,
                                                   train_ann_lookup))
        val_dataset.add_images(image_csv_to_dict(val_img_path, img_style,
                                                 val_ann_lookup))
        train_dataset.add_categories(categories)
        val_dataset.add_categories(categories)

        train_dataset.add_annotations(train_ann_fps)
        val_dataset.add_annotations(val_ann_fps)

        name_prefix = 'deepscores'

        print('\nWriting training annotation file to disk...')
        train_dataset.output_json(osp.join(
            arguments.DIR, name_prefix + '_train.json')
        )

        print('\nWriting test annotation file to disk...')
        val_dataset.output_json(osp.join(
            arguments.DIR, name_prefix + '_test.json')
        )

    else:
        name_prefix = 'deepscores'
        num_train_datasets = num_train_imgs // 2000 + 1
        num_val_datasets = num_test_imgs // 2000 + 1

        def generate_full_datasets(num_datasets: int, img_path: str,
                                   ann_lookup: dict, desc: str,
                                   ann_fps: list, train_dataset: bool):
            images = image_csv_to_dict(img_path, img_style, ann_lookup)
            fp_idx = 0

            for i in range(num_datasets):
                dataset = OBBAnnotations(desc, version='2.0',
                                         subset=f'complete-{i}')
                if i != (num_datasets - 1):
                    dataset.add_images(images[2000 * i:2000 * (i + 1)])
                else:
                    dataset.add_images(images[2000 * i:])
                dataset.add_categories(categories)
                fp_idx = dataset.add_annotations(ann_fps, fp_idx)

                if train_dataset:
                    print(f'\nWriting training annotation file {i} to disk...')
                    dataset.output_json(osp.join(
                        dir_path, name_prefix + f'-complete-{i}_train.json'))
                else:
                    print(f'\nWriting test annotation file {i} to '
                          f'disk...')
                    dataset.output_json(osp.join(
                        dir_path, name_prefix + f'-complete-{i}_test.json'))
            del images
            del dataset

        generate_full_datasets(num_train_datasets, train_img_path,
                               train_ann_lookup, train_desc, train_ann_fps,
                               True)
        generate_full_datasets(num_val_datasets, val_img_path,
                               val_ann_lookup, val_desc, val_ann_fps,
                               False)


if __name__ == '__main__':
    arguments = parse_argument()
    if arguments.CLASSES == '':
        classes = Path(__file__).parent.parent.absolute()
        classes = classes / 'dataset' / 'extended_class_names.csv'
    else:
        classes = Path(arguments.CLASSES)
    start_time = time()

    do_conversion(arguments.DIR, classes, arguments.dense, arguments.y)

    print('\nConversion completed!')
    print(f"Total time: {pretty_time_delta(time() - start_time)}")
