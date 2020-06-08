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
from os.path import join, splitext
import numpy as np
from PIL import Image
from math import floor, ceil, isnan
from itertools import groupby
from tqdm import tqdm
from cv2 import minAreaRect, boxPoints
import pandas as pd


# Some objects are not processed by the mask generation code. To deal with
# these, we'll simply have a set of these objects. If the name is in this set,
# then the segmentation will simply be the entire aligned bbox area.
NON_COLORED_NAMES = {'staffLine'}
OFF_BY_ONE = {'gClef', 'fClef'}
RENAMED_MAPPINGS = {'dynamicF': 'dynamicForte',
                    'dynamicP': 'dynamicPiano',
                    'dynamicM': 'dynamicMezzo',
                    'combStaff': 'staffLine'}


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
    xmin = floor(bbox[0])
    ymin = floor(bbox[1])
    xmax = ceil((bbox[0] + bbox[2]))
    ymax = ceil((bbox[1] + bbox[3]))

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


def generate_oriented_annotations(pix_annotations_dir: str,
                                  xml_annotations_dir: str,
                                  categories: pd.DataFrame,
                                  img_lookup: dict, train_set: set,
                                  val_set: set = None) -> tuple:
    """Generates OBB annotations only.

    Args:
        pix_annotations_dir: Path to the pix_annotations directory,
            i.e. where the segmented images are.
        xml_annotations_dir: Path to the xml_annotations directory,
            i.e. where the object annotations are.
        categories: DataFrame containing the information in the class_names csv
            file.
        img_lookup: A lookup table to get the image_id of every named image.
        train_set: A set that includes the image names of every image in the
            training set. Used to separate which annotation list the
            annotation should be appended to.
        val_set: A set that includes the names of every image in the validation
            set. If None is given, then it is assumed every file not in the
            training set are part of the validation set.

    Returns:
        A tuple of annotation lists, the 0th element being the training ann
        dict, the 1st being the validation ann dict, the 2nd being the a dict of
        annotations with training image ids as the key, and the 3rd being a dict
        of annotations with validation image ids as the key.
    """
    print("Processing annotations...")
    train_ann_list = dict()
    val_ann_list = dict()
    train_ann_lookup = dict()
    val_ann_lookup = dict()

    # Change index of categories to use the xml name
    cat_set = set(categories['xml_name'].to_list())
    categories.set_index('xml_name', inplace=True)

    counter = 1

    file_list = listdir(xml_annotations_dir)

    for file_name in tqdm(file_list):
        img_name = splitext(file_name)[0]

        img_in_train = img_name in train_set
        if val_set is not None:
            img_in_val = img_name in val_set
        else:
            img_in_val = not img_in_train

        if not (img_in_val or img_in_train):
            continue

        xml_path = join(xml_annotations_dir, file_name)
        segmentation_path = join(pix_annotations_dir, img_name + '_seg.png')
        file_annotations = []
        image_id = img_lookup[img_name]

        seg_array = np.array(Image.open(segmentation_path))
        # NOTE: seg_array.shape = (height, width)

        # Generate tree from xml
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        # Get width and height to use as multipliers
        w = float(size.find('width').text)
        h = float(size.find('height').text)

        for obj in root.iter('object'):
            name = obj.find('name').text
            if name in cat_set:
                # Get information about this category
                cat = categories.loc[name]
                ds_cat = str(cat['deepscores_category_id'])
                if isnan(cat['muscima_id']):
                    muscima_cat = None
                else:
                    muscima_cat = str(int(cat['muscima_id']))

                # Get the abs aligned bounding box of the specific annotation
                aligned_bbox = get_aligned_bbox(obj.find('bndbox'), h, w)

                # Get the segmentation
                extracted_seg = extract_area_inside_bbox(aligned_bbox,
                                                         seg_array)

                class_color = int(cat['id'])
                bin_mask, area = generate_binary_mask(extracted_seg,
                                                      class_color)

                if area == 0:
                    continue

                oriented_bbox = get_oriented_bbox(aligned_bbox, bin_mask)

                # Convert aligned_bbox to the right values, i.e. [x0,y0,x1,y1]
                aligned_bbox[2] += aligned_bbox[0]
                aligned_bbox[3] += aligned_bbox[1]
                curr_ann = {
                    'a_bbox': aligned_bbox,
                    'o_bbox': oriented_bbox.tolist(),
                    'cat_id': [ds_cat,
                               muscima_cat],
                    'area': area,
                    'img_id': str(image_id),
                    'comments': ""
                }
                if img_in_train:
                    train_ann_list[str(counter)] = curr_ann
                else:
                    val_ann_list[str(counter)] = curr_ann
                file_annotations.append(str(counter))
                counter += 1
        if img_in_train:
            train_ann_lookup[image_id] = file_annotations
        elif img_in_val:
            val_ann_lookup[image_id] = file_annotations

    return train_ann_list, val_ann_list, train_ann_lookup, val_ann_lookup


def generate_annotations(pix_annotations_dir: str, xml_annotations_dir: str,
                         img_lookup: dict, train_set: set,
                         category_lookup: dict = None, val_set: set = None,
                         oriented: bool = False) -> tuple:
    """Generates COCO-like annotations.

    Args:
        pix_annotations_dir: Path to the pix_annotations directory,
            i.e. where the segmented images are.
        xml_annotations_dir: Path to the xml_annotations directory,
            i.e. where the object annotations are.
        category_lookup: A lookup table to get the category ID of every named
            category.
        img_lookup: A lookup table to get the image_id of every named image.
        train_set: A set that includes the image names of every image in the
            training set. Used to separate which annotation list the
            annotation should be appended to.
        val_set: A set that includes the names of every image in the validation
            set. If None is given, then it is assumed every file not in the
            training set are part of the validation set.
        oriented: Whether or not to use the oriented bounding box schema.t

    Returns:
        A tuple of annotation lists, the 0th element being the training list,
        the 1st being the validation list, the 2nd being the list of
        annotations per training image, and the 3rd being the list of
        annotations per validation image.
    """
    if category_lookup is None:
        assert not oriented, 'Category lookup dictionary is required.'
    print("Processing annotations...")
    train_annotation_list = {} if oriented else []
    test_annotation_list = {} if oriented else []
    train_annotation_lookup = {}
    test_annotation_lookup = {}
    broken_names = set()

    counter = 1

    file_list = listdir(xml_annotations_dir)

    for file_name in tqdm(file_list):
        img_name = splitext(file_name)[0]

        img_in_train = img_name in train_set
        if val_set is not None:
            img_in_val = img_name in val_set
        else:
            img_in_val = not img_in_train

        if not (img_in_val or img_in_train):
            continue

        xml_path = join(xml_annotations_dir, file_name)
        segmentation_path = join(pix_annotations_dir, img_name + '_seg.png')
        file_annotations = []
        image_id = img_lookup[img_name]

        seg_array = np.array(Image.open(segmentation_path))
        # NOTE: seg_array.shape = (height, width)

        # Generate tree from xml
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        # Get width and height to use as multipliers
        w = float(size.find('width').text)
        h = float(size.find('height').text)

        for obj in root.iter('object'):
            # Go through each annotation
            name = obj.find('name').text
            if 'notehead' in name:
                if int(obj.find('rel_position').text) % 2 == 0:
                    name += 'Online'
                else:
                    name += 'Offline'
            if name in RENAMED_MAPPINGS:
                name = RENAMED_MAPPINGS[name]
            if name in category_lookup:
                # First get category ID
                if "change" in name:
                    # NOTE: This branch is now deprecated.
                    pass
                else:
                    category_id = category_lookup[name]

                # Second, get aligned bbox
                aligned_bbox = get_aligned_bbox(obj.find('bndbox'), h, w)

                # Then calculate segmentation
                # Extract the bbox area from the segmentation
                extracted_seg = extract_area_inside_bbox(aligned_bbox,
                                                         seg_array)

                # Then turn it into a binary mask
                class_color = int(category_id)

                if name in NON_COLORED_NAMES:
                    bin_mask = np.ones_like(extracted_seg)
                    area = bin_mask.size
                else:
                    bin_mask, area = generate_binary_mask(extracted_seg,
                                                          class_color)

                if area == 0:
                    continue
                if not oriented:
                    rle_segmentation = binary_mask_to_rle(bin_mask)
                    annotation = {
                        'segmentation': rle_segmentation,
                        'area': area,
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': aligned_bbox,
                        'category_id': category_id,
                        'id': counter
                    }
                    if img_in_train:
                        train_annotation_list.append(annotation)
                    elif img_in_val:
                        test_annotation_list.append(annotation)
                else:
                    if area == 0:
                        bin_mask = np.ones_like(bin_mask)
                    oriented_bbox = get_oriented_bbox(aligned_bbox, bin_mask)
                    curr_ann = {
                        'a_bbox': aligned_bbox,
                        'o_bbox': oriented_bbox.tolist(),
                        'cat_id': category_id,
                        'area': area,
                        'img_id': str(image_id)
                    }
                    if img_in_train:
                        train_annotation_list[str(counter)] = curr_ann
                    elif img_in_val:
                        test_annotation_list[str(counter)] = curr_ann
                file_annotations.append(str(counter))
                counter += 1
            else:
                broken_names.add(name)
        if img_in_train:
            train_annotation_lookup[image_id] = file_annotations
        elif img_in_val:
            test_annotation_lookup[image_id] = file_annotations

    return (train_annotation_list, test_annotation_list,
            train_annotation_lookup, test_annotation_lookup)


def get_aligned_bbox(bndbox, h, w):
    """Gets the axis aligned absolute bounding box.

    This function gets the aligned bounding box of an object based on the
    bndbox value given in the xml description of the object. h and w are used
    to change the values from relative to absolute.
    """
    xmin = float(bndbox.find('xmin').text) * w
    ymin = float(bndbox.find('ymin').text) * h
    width = (float(bndbox.find('xmax').text) * w) - xmin
    height = (float(bndbox.find('ymax').text) * h) - ymin

    return [xmin, ymin, round(width, 8), round(height, 8)]


def get_oriented_bbox(aligned_bbox, bin_mask):
    """Calculates the oriented bounding box around an object.

    Returns:
        np.ndarray: bbox as a (8,) numpy array
    """
    adders = np.array(aligned_bbox[0:2])

    bin_indices = np.transpose(np.nonzero(bin_mask))
    min_box = boxPoints(minAreaRect(bin_indices))
    min_box = np.flip(min_box, 1)  # For some reason this is in the wrong order
    min_box += adders  # shift them to their actual absolute position

    return np.concatenate(min_box)
