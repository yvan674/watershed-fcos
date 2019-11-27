"""Images.

Processes the images in the images_png directory.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 18, 2019
"""
from os import listdir
from os.path import join, exists
from PIL import Image
import csv

Image.MAX_IMAGE_PIXELS = None


def process_image_dir(dir: str, work_dir: str, training_set: set) -> tuple:
    """Processes dir to produce a COCO-like image list and a lookup table.

    Args:
        dir: Directory with images.
        work_dir: Temporary directory to save csv files too.
        training_set: Set of files that should be considered training data.

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
                img_name = image.split('.')[0]
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

    for image in dir_list:
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
        img_name = image.split('.')[0]
        lookup_table[img_name] = counter

        # Append to the appropriate list
        if img_name in training_set:
            train_list.append(data)
        else:
            val_list.append(data)

        if counter % 50 == 0 or counter == len(dir_list):
            print('Processing image {} of {}'.format(counter, len(dir_list)))

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

    print("Done processing images!")

    return join(work_dir, 'training_list.csv'), \
           join(work_dir, 'validation_list.csv'), \
           lookup_table
