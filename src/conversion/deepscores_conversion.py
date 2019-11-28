"""DeepScores conversion.

Converts DeepScores styled annotations to COCO style annotations.

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
    parser.add_argument('-x', '--extended', action='store_true',
                        help='sets the conversion to convert the extended '
                             'dataset. This requires the name of the '
                             'class_names file to be class_names_extended.csv')

    return parser.parse_args()


def process_class_colors(file_path: str) -> dict:
    """Turns the class_names csv to a dict """
    class_name_dict = dict()
    with open(file_path, 'r') as class_names:
        reader = csv.reader(class_names)
        for line in reader:
            class_name_dict[line[1]] = line[0]

    return class_name_dict


def do_conversion(dir_path: str, extended: bool) -> tuple:
    """Does the actual conversion.

    Returns:
        A tuple where the first element is the training CocoLikeAnnotations
        and the second element is the validation CocoLikeAnnotations.
    """
    training_set = set()

    # Get the training set and validation split as sets
    with open(join(dir_path, 'train_names.csv')) as train_names:
        reader = csv.reader(train_names)
        for line in reader:
            training_set.add(line[1])

    # Make the work dir if it doesn't exist
    work_dir = join(dir_path, 'tmp')
    if not exists(work_dir):
        mkdir(work_dir)

    train_img_path, val_img_path, img_lookup = \
        process_image_dir(join(dir_path, 'images_png'), work_dir, training_set)

    # This is quick so we don't need to save results to disk
    if extended:
        cat_file_name = 'class_names_extended.csv'
    else:
        cat_file_name = 'class_names.csv'
    categories, cat_lookup = generate_categories(join(dir_path,
                                                      cat_file_name))

    class_colors = process_class_colors(join(dir_path, cat_file_name))
    train_ann, val_ann = generate_annotations(
        pix_annotations_dir=join(dir_path, 'pix_annotations_png'),
        xml_annotations_dir=join(dir_path, 'xml_annotations'),
        category_lookup=cat_lookup,
        img_lookup=img_lookup,
        class_colors=class_colors,
        train_set=training_set,
        work_dir=join(dir_path))

    desc = "DeepScores as COCO Dataset"
    return (CocoLikeAnnotations(desc, train_img_path, categories,
                                train_ann),
            CocoLikeAnnotations(desc, val_img_path, categories, val_ann))


if __name__ == '__main__':
    arguments = parse_argument()

    dir_path = arguments.DIR[0]

    converted_train, converted_val = do_conversion(dir_path,
                                                   arguments.extended)
    print('\nWriting training annotation file to disk...')
    converted_train.output_json(join(dir_path, 'deepscores_train.json'))

    print('\nWriting validation annotation file to disk...')
    converted_val.output_json(join(dir_path, 'deepscores_val.json'))
    print('\nConversion completed!')
