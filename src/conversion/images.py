"""Images.

Processes the images in the images_png directory.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 18, 2019
"""
from os import listdir
from os.path import join
from PIL import Image


def process_image_dir(dir: str):
    """Processes dir to produce a COCO-like image list and a lookup table."""
    img_list = []
    lookup_table = dict()
    counter = 1
    dir_listdir = listdir(dir)
    len_dir_listdir = len(dir_listdir)
    for image in dir_listdir:
        img_file = Image.open(join(dir, image))
        width, height = img_file.size
        img_list.append({
            'license': 1,
            'file_name': image,
            'coco_url': 'not_a_coco_image',
            'height': height,
            'width': width,
            'date_captured': '1970-01-01 00:00:00',
            'flickr_url': 'not_on_flickr',
            'id': counter
        })
        lookup_table[image.split('.')[0]] = counter

        if counter % 50 == 0 or counter == len_dir_listdir:
            print('Processing image {} of {}'.format(counter, len_dir_listdir))
        counter += 1

    return img_list, lookup_table
