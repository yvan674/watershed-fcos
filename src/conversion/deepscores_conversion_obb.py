"""Deepscores Converstion to Oriented Bounding Box Annotations.

Converts the deepscores dataset to the Oriented Bounding Box Annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    March 03, 2020
"""
import os.path as osp
import os
from conversion import OBBAnnotations, process_image_dir, \
    generate_oriented_categories, generate_oriented_annotations, \
    image_csv_to_dict
from conversion.deepscores_conversion import process_class_colors
import csv
import argparse


def parse_argument():
    desc = 'converts the DeepScores dataset to conform to the Oriented ' \
           'BoundingBox dataset schema.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('DIR', type=str,
                        help='path to the directory containing the dataset')
    parser.add_argument('CLASSES', type=str,
                        help='path to the file containing the class names list')

    return parser.parse_args()


def do_conversion(dir_path: str, class_names_fp: str) -> tuple:
    """Does the actual conversion.

    Returns:
        A tuple where the first element is the training OBBAnnotations
        and the second element is the validation OBBAnnotations.
    """
    training_set = set()

    # Get the training set and validation split as sets
    with open(osp.join(dir_path, 'train_names.csv')) as train_names:
        reader = csv.reader(train_names)
        for line in reader:
            training_set.add(line[1])

    # Make sure the work directory exists
    work_dir = osp.join(dir_path, 'tmp')
    if not osp.exists(work_dir):
        os.mkdir(work_dir)

    train_img_path, val_img_path, img_lookup = \
        process_image_dir(osp.join(dir_path, 'images_png'), work_dir,
                          training_set)

    categories, cat_lookup, cat_set = generate_oriented_categories(osp.join(
        dir_path, class_names_fp
    ))

    class_colors = process_class_colors(osp.join(dir_path, class_names_fp))
    train_ann, val_ann, train_ann_lookup, val_ann_lookup = \
    generate_oriented_annotations(
        pix_annotations_dir=osp.join(dir_path, 'pix_annotations_png'),
        xml_annotations_dir=osp.join(dir_path, 'xml_annotations'),
        category_lookup=cat_lookup,
        img_lookup=img_lookup,
        class_colors=class_colors,
        train_set=training_set,
        category_set=cat_set
    )

    # Once that's complete, generate the actual dataset objects.
    train_dataset = OBBAnnotations("DeepScores training set as an OBB Dataset")
    val_dataset = OBBAnnotations("DeepScores validation set as an OBB dataset")

    train_dataset.add_images(image_csv_to_dict(train_img_path, 'obb',
                                               train_ann_lookup))
    val_dataset.add_images(image_csv_to_dict(val_img_path, 'obb',
                                             val_ann_lookup))

    train_dataset.add_categories(categories)
    val_dataset.add_categories(categories)

    train_dataset.add_annotations(train_ann)
    val_dataset.add_annotations(val_ann)

    return train_dataset, val_dataset

if __name__ == '__main__':
    arguments = parse_argument()

    train, val = do_conversion(arguments.DIR, arguments.CLASS)

    print('\nWriting training annotation file to disk...')
    train.output_json(osp.join(arguments.DIR, 'deepscores_oriented_train.json'))

    print('\nWriting validation annotation file to disk...')
    val.output_json(osp.join(arguments.DIR, 'deepscores_oriented_val.json'))

    print('\nConversion completed!')
