"""Images.

Processes the images in the images_png directory.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 18, 2019
"""
from os import listdir
from os.path import join, exists, splitext
from PIL import Image
import csv

from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def process_image_dir(dir: str, work_dir: str, training_set: set,
                      val_set: set = None) -> tuple:
    """Processes dir to produce a COCO-like image list and a lookup table.

    Args:
        dir: Directory with images.
        work_dir: Temporary directory to save csv files too.
        training_set: Set of files that should be considered training data.
        val_set: Set of files that should be considered validation data. If None
            is given, then it is assumed any file not in the training set is
            part of the validation set.

    Returns:
        A tuple containing the path to the training csv (str),
        validation csv (str), and image id lookup table (dict).
    """
    counter = 1
    lookup_table = dict()
    dir_list = listdir(dir)

    train_fp = join(work_dir, 'training_list.csv')
    val_fp = join(work_dir, 'validation_list.csv')
    lookup_fp = join(work_dir, 'img_lookup.csv')

    # Check if the CSVs already exists and if it does, don't compute.
    if exists(train_fp) and exists(val_fp) and exists(lookup_fp):
        print('CSV files exist, reading lookup file...')
        with open(lookup_fp, mode='r') as lookup_file:
            reader = csv.reader(lookup_file)
            for row in reader:
                lookup_table[row[0]] = int(row[1])
        print('Done!')
        return train_fp, val_fp, lookup_table

    elif exists(train_fp) and exists(val_fp) and not exists(lookup_fp):
        print('CSV files exist, but no lookup table. Generating only the '
              'lookup table CSV file...')
        with open(lookup_fp, mode='w') as lookup_file:
            writer = csv.writer(lookup_file)
            for image in dir_list:
                img_name = splitext(image)[0]
                lookup_table[img_name] = counter
                counter += 1
            writer.writerows(lookup_table.items())
        print('Done!')
        return train_fp, val_fp, lookup_table

    train_file = open(train_fp, mode='w')
    val_file = open(val_fp, mode='w')
    img_lookup_file = open(lookup_fp, mode='w')

    fields = ['license', 'file_name', 'coco_url', 'height', 'width',
              'date_captured', 'flickr_url', 'id']
    train_writer = csv.DictWriter(train_file, fields)
    train_writer.writeheader()
    val_writer = csv.DictWriter(val_file, fields)
    val_writer.writeheader()

    train_list = []
    val_list = []

    for image in tqdm(dir_list):
        img_file = Image.open(join(dir, image))
        width, height = img_file.size
        data = {
            'license': 1,
            'file_name': image,
            'coco_url': 'not_a_coco_image',
            'height': height,
            'width': width,
            'date_captured': '1970-01-01 00:00:00',
            'flickr_url': 'not_on_flickr',
            'id': counter
        }
        img_name = splitext(image)[0]
        lookup_table[img_name] = counter

        # Append to the appropriate list
        if img_name in training_set:
            train_list.append(data)
        else:
            if val_set is None:
                val_list.append(data)
            elif img_name in val_set:
                val_list.append(data)

        if counter % 500 == 0 or counter == len(dir_list):
            train_writer.writerows(train_list)
            val_writer.writerows(val_list)
            train_list = []
            val_list = []

        counter += 1

    # Write out lookup file
    print("Writing lookup file...")
    img_lookup_writer = csv.writer(img_lookup_file)
    img_lookup_writer.writerows(lookup_table.items())

    # Close out files when we finish
    train_file.close()
    val_file.close()
    img_lookup_file.close()

    print("Done processing images!")

    return join(work_dir, 'training_list.csv'), \
           join(work_dir, 'validation_list.csv'), \
           lookup_table


def image_csv_to_dict(fp:str, style:str, ann_lookup:dict=None) -> list:
    """Reads the image csv file and turns it into a list of dictionaries.

    Args:
        fp: Path to the csv file.
        style: Which style to return. Possible options are 'coco', 'obb'
        train_ann_lookup: Training annotations lookup.
        val_ann_lookup: Validation annotations lookup.

    Returns:
        The list of images as dictionaries according to the required style.
    """
    assert style in ('coco', 'obb'), 'The chosen style is not implemented.'
    print('Reading image CSV file...')
    image_list = []
    with open(fp, mode='r') as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        for row in reader:
            if style is 'coco':
                image_list.append({
                    'license': int(row['license']),
                    'file_name': row['file_name'],
                    'coco_url': row['coco_url'],
                    'height': int(row['height']),
                    'width': int(row['width']),
                    'date_captured': row['date_captured'],
                    'flickr_url': row['flickr_url'],
                    'id': int(row['id'])
                })
            elif style is 'obb':
                img_id = int(row['id'])
                image_list.append({
                    'id': img_id,
                    'filename': row['file_name'],
                    'width': int(row['width']),
                    'height': int(row['height']),
                    'ann_ids': ann_lookup[img_id]
                })
    return image_list
