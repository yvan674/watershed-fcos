"""Annotations

Produces COCO compatible annotations for the DeepScores dataset.

References:
    Binary mask to RLE encoding implementation from
    <https://github.com/waspinator/pycococreator/>, licensed under the Apache
    License 2.0. No changes were made.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 18, 2019
"""
from xml.etree import ElementTree as ET
from os import listdir
from os.path import join
import numpy as np
from PIL import Image
from math import floor, ceil
from itertools import groupby


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(
            binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle


def extract_area_inside_bbox(bbox: list,
                             segmentation_mask: np.ndarray) -> np.ndarray:
    """Extracts just the area inside the bounding box from the seg mask."""
    height, width = segmentation_mask.shape

    xmin = floor(bbox[0] * width)
    ymin = floor(bbox[1] * height)
    xmax = ceil((bbox[0] + bbox[2]) * width)
    ymax = ceil((bbox[1] + bbox[3]) * height)

    extracted_mask = segmentation_mask[ymin:ymax, xmin:xmax]
    return np.asfortranarray(extracted_mask)


def generate_binary_mask(seg_mask: np.ndarray, category_code: int) -> tuple:
    """Generates a binary mask of a single category.

    Args:
        seg_mask: The extracted segmentation mask of only the area inside the
            bounding box
        category_code: The gray value of the specific class that is to be
            extracted. This should be taken from class_names.csv

    Returns:
        A tuple with element 1 being the binary mask and element 2 being the
        area of the mask.
    """
    bin_mask = seg_mask == category_code
    return bin_mask.astype('uint8'), np.count_nonzero(bin_mask)


def generate_annotations(pix_annotations_dir: str, xml_annotations_dir: str,
                         category_lookup: dict, img_lookup: dict,
                         class_colors: dict) -> list:
    """Generates COCO-like annotations."""
    annotation_list = []
    counter = 1

    file_list = listdir(xml_annotations_dir)
    len_file_list = len(file_list)
    file_counter = 1

    for file_name in file_list:
        xml_path = join(xml_annotations_dir, file_name)
        segmentation_path = join(pix_annotations_dir, file_name.split('.')[0]
                                 + '.png')

        seg_array = np.array(Image.open(segmentation_path))
        # NOTE: seg_array.shape = (height, width)

        # Generate tree from xml
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            # Go through each annotation
            name = obj.find('name').text
            if name != "brace":
                # Get bounding box values
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                width = float(bndbox.find('xmax').text) - xmin
                height = float(bndbox.find('ymax').text) - ymin

                bbox = [xmin, ymin, round(width, 8), round(height, 8)]

                # Get category ID
                category_id = category_lookup[name]

                # Calculate segmentation
                # First extract the bbox area from the segmentation
                extracted_seg = extract_area_inside_bbox(bbox, seg_array)

                # Then turn it into a binary mask
                class_color = int(class_colors[name])
                bin_mask, area = generate_binary_mask(extracted_seg,
                                                      class_color)
                rle_segmentation = binary_mask_to_rle(bin_mask)

                annotation_list.append({
                    'segmentation': rle_segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': img_lookup[file_name.split('.')[0]],
                    'bbox': bbox,
                    'category_id': category_id,
                    'id': counter
                })
                counter += 1
        if file_counter % 50 == 0 or file_counter == len_file_list:
            print('Processed {} of {} xml files'.format(file_counter,
                                                        len_file_list))
        file_counter += 1
    return annotation_list
