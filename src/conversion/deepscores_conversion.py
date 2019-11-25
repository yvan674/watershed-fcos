"""DeepScores conversion.

Converts DeepScores styled annotations to COCO style annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import sys
from os.path import join
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

    Note:
        This is not the mos efficient way to do this, but we need to do this
        only rarely so it doesn't matter. A more efficient way would be to
        split it into the training and validation sets as we go but that
        requires changing how the images are processed.

    Returns:
        A tuple where the first element is the training CocoLikeAnnotations
        and the second element is the validation CocoLikeAnnotations.
    """
    training_set = set()
    validation_set = set()

    # Get the training set and validation set as sets
    with open(join(dir_path, 'train_names.csv')) as train_names:
        reader = csv.reader(train_names)
        for line in reader:
            training_set.add(line[1])

    with open(join(dir_path, 'test_names.csv')) as test_names:
        reader = csv.reader(test_names)
        for line in reader:
            validation_set.add(line[1])

    images, img_lookup = process_image_dir(join(dir_path, 'images_png'))
    categories, cat_lookup = generate_categories(join(dir_path,
                                                      'class_names.csv'))

    class_names = process_class_names(dir_path)
    annotations = generate_annotations(join(dir_path, 'pix_annotations_png'),
                                       join(dir_path, 'xml_annotations'),
                                       cat_lookup, img_lookup,
                                       class_names)

    # Now split the image into validation and training
    print('Splitting images...')
    train_images = []
    train_ids = set()
    val_images = []
    val_ids = set()
    for image in images:
        name = image['file_name'].split('.')[0]

        if name in training_set:
            train_images.append(image)
            train_ids.add(image['id'])
        elif name in validation_set:
            val_images.append(image)
            val_ids.add(image['id'])

    print('Splitting annotations...')
    # And now split the annotations into validation and training
    train_annotations = []
    val_annotations = []
    for annotation in annotations:
        img_id = annotation['image_id']
        if img_id in train_ids:
            train_annotations.append(annotation)
        elif img_id in val_ids:
            val_annotations.append(annotation)

    desc = "DeepScores as COCO Dataset"
    return (CocoLikeAnnotations(desc, train_images, categories,
                                train_annotations),
            CocoLikeAnnotations(desc, val_images, categories, val_annotations))


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
