"""DeepScores conversion.

Converts DeepScores styled annotations to COCO style annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import sys
from os.path import join, exists
from os import mkdir
from conversion.coco_annotations import CocoLikeAnnotations
from conversion.images import process_image_dir
from conversion.categories import generate_categories
from conversion.annotations import generate_annotations
import csv


def process_class_names(dir_path: str) -> dict:
    """Turns the class_names csv to a dict """
    class_name_dict = dict()
    with open(join(dir_path, 'class_names.csv'), 'r') as class_names:
        reader = csv.reader(class_names)
        for line in reader:
            class_name_dict[line[1]] = line[0]

    return class_name_dict


def do_conversion(dir_path: str) -> tuple:
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
    categories, cat_lookup = generate_categories(join(dir_path,
                                                      'class_names.csv'))

    class_names = process_class_names(dir_path)
    train_ann, val_ann = generate_annotations(
        pix_annotations_dir=join(dir_path, 'pix_annotations_png'),
        xml_annotations_dir=join(dir_path, 'xml_annotations'),
        category_lookup=cat_lookup,
        img_lookup=img_lookup,
        class_colors=class_names,
        train_set=training_set,
        work_dir=join(dir_path))

    desc = "DeepScores as COCO Dataset"
    return (CocoLikeAnnotations(desc, train_img_path, categories,
                                train_ann),
            CocoLikeAnnotations(desc, val_img_path, categories, val_ann))


if __name__ == '__main__':
    try:
        dir_path = sys.argv[1]
    except IndexError:
        print("Requires a directory path")
        sys.exit(1)

    converted_train, converted_val = do_conversion(dir_path)
    print('Writing training annotation file to disk...')
    converted_train.output_json(join(dir_path, 'deepscores_train.json'))

    print('Writing validation annotation file to disk...')
    converted_val.output_json(join(dir_path, 'deepscores_val.json'))
    print('Conversion completed!')
