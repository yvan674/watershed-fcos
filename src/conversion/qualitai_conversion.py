"""QualitAI conversion.

Converts QualitAI styled annotations to COCO style annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from os.path import join, split
from os import listdir
import argparse
from xml.etree import ElementTree as ET
import numpy as np
from datetime import datetime as dt
from PIL import Image
import json
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser(description='converts the QualitAI '
                                                 'dataset into a COCO-like '
                                                 'dataset')

    parser.add_argument('DIR', type=str, nargs=1,
                        help='path to directory containing the dataset')
    parser.add_argument('-a', '--all', action='store_true',
                        help='does not differentiate between bad and good '
                             'samples')

    return parser.parse_args()


def generate_annotations(directory: str, combine_sets: bool) -> tuple:
    """Generates coco-like annotations from a QualitAI dataset.

    Args:
        directory: The image directory to process. If combine_sets is false,
            then this is the absolute path to 'train/good' and so on. If
            combine_sets is true, then this is just the absolute path to
            'train'.
        combine_sets: Whether or not to combine the good and bad samples.

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
    bad_images = []
    desc = "Processing folder {}".format(split(directory)[-1])

    # Get the full file lists
    if combine_sets:
        file_list = [join('good', file)
                     for file in listdir(join(directory, 'good'))]
        file_list.extend([join('bad', file)
                          for file in listdir(join(directory, 'bad'))])
    else:
        file_list = listdir(directory)

    # Make an image name list which doesn't contain extensions
    for file_name in file_list:
        if "orig.png" in file_name:
            image_names.append(file_name.split('.')[0])

    if combine_sets:
        for img_name in tqdm(image_names, desc):
            # ins stands for img_name_split and is used for the
            # date_captured info
            ins = img_name.split('_')

            date_captured = "{}-{}-{} {}:{}:{}".format(*ins)
            if 'bad' in img_name:
                # Only parse annotations if it's not part of the good set.
                ann_tree = ET.parse(join(directory, img_name + '_bboxes.xml'))
                seg_tree = ET.parse(join(directory, img_name + '.xml'))

                ann_root = ann_tree.getroot()
                seg_root = seg_tree.getroot()

                w = float(ann_root.find('size').find('width').text)
                h = float(ann_root.find('size').find('height').text)

                for seg, obj in zip(seg_root.iter('Stroke'),
                                    ann_root.iter('object')):
                    # Go through each bounding box in the image
                    # First do segmentations
                    segmentation = []
                    x_coords = []
                    y_coords = []
                    for point in seg.iter('Point'):
                        segmentation.append(float(point.attrib['X']))
                        x_coords.append(float(point.attrib['X']))

                        segmentation.append(float(point.attrib['Y']))
                        y_coords.append(float(point.attrib['Y']))

                    # Then calculate its area
                    x_coords = np.array(x_coords)
                    y_coords = np.array(y_coords)
                    area = 0.5 * np.abs(np.dot(x_coords,
                                               np.roll(y_coords, 1))
                                        - np.dot(y_coords,
                                                 np.roll(x_coords, 1)))
                    # Now calculate the bounding box
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text) * w
                    ymin = float(bndbox.find('ymin').text) * h
                    width = (float(bndbox.find('xmax').text) * w) - xmin
                    height = (float(bndbox.find('ymax').text) * h) - ymin

                    bbox = [xmin, ymin, round(width, 8), round(height, 8)]

                    if not bbox[2] == 0 and not bbox[3] == 0:
                        ann_list.append({
                            'segmentation': [segmentation],
                            'area': area,
                            'iscrowd': 0,
                            'image_id': img_counter,
                            'bbox': bbox,
                            'category_id': 1,
                            'id': ann_counter
                        })
                        ann_counter += 1
                    else:
                        bad_images.append(img_name)

            else:
                w, h = Image.open(join(directory, img_name + '.png')).size

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
            img_counter += 1

    else:
        if 'good' in directory:
            # Then there aren't any xml files and we can just return no
            # annotations, but we have to go through each image to check its
            # height
            for img_name in tqdm(image_names, desc):
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
                img_counter += 1

        else:
            # Iterate through every image
            for img_name in tqdm(image_names, desc):
                image_appended = False
                # First parse the xml files we need
                ann_tree = ET.parse(join(directory, img_name + '_bboxes.xml'))
                seg_tree = ET.parse(join(directory, img_name + '.xml'))

                ann_root = ann_tree.getroot()
                seg_root = seg_tree.getroot()

                w = float(ann_root.find('size').find('width').text)
                h = float(ann_root.find('size').find('height').text)

                ins = img_name.split('_')  # ins stands for img_name_split

                date_captured = "{}-{}-{} {}:{}:{}".format(*ins)

                for seg, obj in zip(seg_root.iter('Stroke'),
                                    ann_root.iter('object')):
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
                    area = 0.5 * np.abs(np.dot(x_coords,
                                               np.roll(y_coords, 1))
                                        - np.dot(y_coords,
                                                 np.roll(x_coords, 1)))

                    # Now calculate the bounding box
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text) * w
                    ymin = float(bndbox.find('ymin').text) * h
                    width = (float(bndbox.find('xmax').text) * w) - xmin
                    height = (float(bndbox.find('ymax').text) * h) - ymin

                    bbox = [xmin, ymin, round(width, 8), round(height, 8)]

                    if not bbox[2] == 0 and not bbox[3] == 0:
                        if not image_appended:
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
                            image_appended = True

                        ann_list.append({
                            'segmentation': [segmentation],
                            'area': area,
                            'iscrowd': 0,
                            'image_id': img_counter,
                            'bbox': bbox,
                            'category_id': 1,
                            'id': ann_counter
                        })
                        ann_counter += 1
                    else:
                        bad_images.append(img_name)
                img_counter += 1

    if len(bad_images) > 0:
        print("\nBad images:")
        for file in bad_images:
            print(file)
    else:
        print("\nNo bad images found.")

    return image_info_list, ann_list


def do_conversion(dir_path: str, combine_sets: bool) -> list:
    """Does the actual conversion.

    Args:
        dir_path: path to the data directory
        combine_sets: Combines the good and the bad images

    Returns:
        A tuple where the first elements are: the good training
        CocoLikeAnnotations, the bad training CocoLikeAnnotations, the good
        validation CocoLikeAnnotations, and the bad validation
        CocoLikeAnnotations.
    """
    # Set the directory paths
    if combine_sets:
        directories = (
            join(dir_path, 'train'),
            join(dir_path, 'valid')

        )
    else:
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
        'year': int(dt.now().strftime('%Y')),
        'contributor': 'QualitAI and the InIT DataLab',
        'date': dt.now().strftime('%Y/%m/%d')
    }

    licenses = [{'url': 'no_url_given', 'id': 1, 'name': 'No license given'}]

    categories = [{'supercategory': 'classification',
                  'id': 1,
                  'name': 'defect'}]

    coco_annotations = []
    for directory in directories:
        images, annotations = generate_annotations(directory, combine_sets)
        coco_annotations.append({
            'info': info,
            'licenses': licenses,
            'images': images,
            'categories': categories,
            'annotations': annotations
        })
        print('{}: {} images'.format(directory, len(images)))

    return coco_annotations


if __name__ == '__main__':
    arguments = parse_argument()

    dir_path = arguments.DIR[0]

    coco_annotations = do_conversion(dir_path, arguments.all)

    if arguments.all:
        file_names = ('training', 'validation')
    else:
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
