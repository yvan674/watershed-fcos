"""QualitAI conversion.

Converts QualitAI styled annotations to COCO style annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from os.path import join
from os import listdir
import argparse
from xml.etree import ElementTree as ET
import numpy as np
from datetime import datetime as dt
from PIL import Image
import json


def parse_argument():
    parser = argparse.ArgumentParser(description='converts the QualitAI '
                                                 'dataset into a COCO-like '
                                                 'dataset')

    parser.add_argument('DIR', type=str, nargs=1,
                        help='path to directory containing the dataset')

    return parser.parse_args()


def generate_annotations(directory: str) -> tuple:
    """Generates coco-like annotations from a QualitAI dataset.

    Returns:
        A tuple with the first item being the image dict list and the second
        item being the annotations dict list
    """
    # Get a list of all root file_names
    image_info_list = []
    ann_list = []
    img_counter = 1
    ann_counter = 1

    image_names = []
    file_list = listdir(directory)
    for file_name in file_list:
        if "orig.png" in file_name:
            image_names.append(file_name.split('.')[0])

    if 'good' in directory:
        # Then there aren't any xml files and we can just return no
        # annotations, but we have to go through each image to check its height
        for img_name in image_names:
            img_file = Image.open(join(directory, img_name + '.png'))
            w, h = img_file.size
            image_info_list.append({
                'license': 1,
                'file_name': img_name + '.png',
                'coco_url': 'no_coco_url',
                'height': h,
                'width': w,
                'date_captured': '1970-01-01 00:00:00',
                'flickr_url': 'not_on_flickr',
                'id': img_counter
            })
            if img_counter % 50 == 0 or img_counter == len(file_list):
                print('Processed {} of {} images'.format(img_counter,
                                                         len(file_list)))
            img_counter += 1

    else:
        # Iterate through every image
        for img_name in image_names:

            # First parse the xml files we need
            ann_tree = ET.parse(join(directory, img_name + '_bboxes.xml'))
            seg_tree = ET.parse(join(directory, img_name + '.xml'))

            ann_root = ann_tree.getroot()
            seg_root = seg_tree.getroot()

            w = float(ann_root.find('size').find('width').text)
            h = float(ann_root.find('size').find('width').text)

            ins = img_name.split('_')  # ins stands for img_name_split

            date_captured = "{}-{}-{} {}:{}:{}".format(ins[0], ins[1], ins[2],
                                                       ins[3], ins[4], ins[5])

            image_info_list.append({
                'license': 1,
                'file_name': img_name + '.png',
                'coco_url': 'no_coco_url',
                'height': h,
                'width': w,
                'date_captured': date_captured,
                'flickr_url': 'no_flickr_url',
                'id': img_counter
            })

            for seg, obj in zip(seg_root.iter('Stroke'), ann_root.iter('object')):
                # Go through each bounding box in the image
                # First get the segmentations
                segmentation = []
                x_coords = []
                y_coords = []
                for point in seg.iter('Point'):
                    segmentation.append(float(point.attrib['X']))
                    x_coords.append(float(point.attrib['X']))

                    segmentation.append(float(point.attrib['Y']))
                    y_coords.append(float(point.attrib['Y']))

                # Then calculate the area
                x_coords = np.array(x_coords)
                y_coords = np.array(y_coords)
                area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1))
                                    - np.dot(y_coords, np.roll(x_coords, 1)))

                # Now calculate the bounding box
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text) * w
                ymin = float(bndbox.find('ymin').text) * h
                width = (float(bndbox.find('xmax').text) * w) - xmin
                height = (float(bndbox.find('ymax').text) * h) - ymin

                bbox = [xmin, ymin, round(width, 8), round(height, 8)]


                ann_list.append({
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': img_counter,
                    'bbox': bbox,
                    'category_id': 1,
                    'id': ann_counter
                })
                img_counter += 1
                ann_counter += 1

            if img_counter % 50 == 0 or img_counter == len(file_list):
                print('Processed {} of {} images'.format(img_counter,
                                                         len(file_list)))

    return image_info_list, ann_list


def do_conversion(dir_path: str) -> list:
    """Does the actual conversion.

    Returns:
        A tuple where the first elements are: the good training
        CocoLikeAnnotations, the bad training CocoLikeAnnotations, the good
        validation CocoLikeAnnotations, and the bad validation
        CocoLikeAnnotations.
    """
    # Set the directory paths
    directories = (
        join(dir_path, 'train', 'good'),
        join(dir_path, 'train', 'bad'),
        join(dir_path, 'valid', 'good'),
        join(dir_path, 'valid', 'bad')
    )

    info = {
        'description': 'QualitAI as COCO Dataset',
        'url': 'no_url_available',
        'version': '1.0',
        'year': 2019,
        'contributor': 'QualitAI and the InIT DataLab',
        'date': dt.now().strftime('%Y/%m/%d')
    }

    licenses = [{'url': 'no_url_given', 'id': 1, 'name': 'No license given'}]

    categories = {'supercategory': 'classification',
                  'id': 1,
                  'name': 'bad'}

    coco_annotations = []
    for directory in directories:
        images, annotations = generate_annotations(directory)
        coco_annotations.append({
            'info': info,
            'licenses': licenses,
            'images': images,
            'categories': categories,
            'annotations': annotations
        })

    return coco_annotations


if __name__ == '__main__':
    arguments = parse_argument()

    dir_path = arguments.DIR[0]

    coco_annotations = do_conversion(dir_path)

    file_names = ('training_good',
                  'training_bad',
                  'validation_good',
                  'validation_bad')

    for file_name, annotation in zip(file_names, coco_annotations):
        print('\nWriting {} annotation file to disk...'.format(file_name))
        fp = join(dir_path, 'qualitai_{}.json'.format(file_name))
        with open(fp, 'w+') as output_file:
            json.dump(annotation, output_file)

    print('\nConversion completed!')
