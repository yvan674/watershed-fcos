"""DeepScores conversion.

Converts DeepScores styled annotations to COCO style annotations.
This is now redundant as the obb script also converts to coco

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from os.path import join, exists
from os import mkdir
from conversion.coco_annotations import CocoLikeAnnotations
from conversion.images import process_image_dir
from conversion.categories import generate_categories
from conversion.annotations import generate_annotations
import csv
import argparse


def parse_argument():
    parser = argparse.ArgumentParser(description='converts the deepscores '
                                                 'dataset into a COCO-like '
                                                 'dataset')

    parser.add_argument('DIR', type=str, nargs=1,
                        help='path to directory containing the dataset')
    parser.add_argument('CLASSES', type=str,
                        help='path to the file containing the class names list')

    return parser.parse_args()


def do_conversion(dir_path: str, class_names_fp: str) -> tuple:
    """Does the actual conversion.

    Returns:
        A tuple where the first element is the training CocoLikeAnnotations
        and the second element is the validation CocoLikeAnnotations.
    """
    training_set = set()
    val_set = set()

    # Get the training set and validation split as sets
    with open(join(dir_path, 'train_names.csv')) as train_names:
        reader = csv.reader(train_names)
        for line in reader:
            training_set.add(line[1])

    try:
        with open(join(dir_path, 'test_names.csv')) as val_names:
            reader = csv.reader(val_names)
            for line in reader:
                val_set.add(line[1])
    except FileNotFoundError:
        print("WARNING: test_names.csv has not been found. Assuming every file "
              "not in train_names.csv is part of the validation set.")
        val_set = None

    # Make the work dir_path if it doesn't exist
    work_dir = join(dir_path, 'tmp')
    if not exists(work_dir):
        mkdir(work_dir)

    train_img_path, val_img_path, img_lookup = \
        process_image_dir(join(dir_path, 'images_png'), work_dir, training_set,
                          val_set)

    # This is quick so we don't need to save results to disk
    categories, cat_lookup, cat_set = generate_categories(join(dir_path,
                                                               class_names_fp))

    train_ann, val_ann = generate_annotations(
        pix_annotations_dir=join(dir_path, 'pix_annotations_png'),
        xml_annotations_dir=join(dir_path, 'xml_annotations'),
        category_lookup=cat_lookup,
        img_lookup=img_lookup,
        train_set=training_set,
        category_set=cat_set)[0:2]

    desc = "DeepScores as COCO Dataset"
    return (CocoLikeAnnotations(desc, train_img_path, categories,
                                train_ann),
            CocoLikeAnnotations(desc, val_img_path, categories, val_ann))


if __name__ == '__main__':
    arguments = parse_argument()

    dir_path = arguments.DIR[0]

    converted_train, converted_val = do_conversion(dir_path,
                                                   arguments.CLASSES)
    print('\nWriting training annotation file to disk...')
    converted_train.output_json(join(dir_path, 'deepscores_train.json'))

    print('\nWriting validation annotation file to disk...')
    converted_val.output_json(join(dir_path, 'deepscores_val.json'))
    print('\nConversion completed!')
