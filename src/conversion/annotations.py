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
from PIL import Image, ImageColor
from math import floor, ceil, isnan
from itertools import groupby
from tqdm import tqdm
from cv2 import minAreaRect, boxPoints
import pandas as pd
import pickle


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
    """Extracts just the area inside the bounding box from the seg mask.

    Returns:
        A numpy array of the segmentation mask. This array is cropped to an
            area based on the bbox expanded by 100 pixels on each side. The 100
            pixel expansion is done since something the bounding boxes are
            weird or wrong.
    """
    # Note: segmentation mask has shape (h, w, c) where c is channels
    # Max and min are used on each value to make sure the bounds we get are
    # within the actual area of the segmentation mask itself.
    h, w, _ = segmentation_mask.shape
    xmin = max(0, floor(bbox[0] - 100))
    ymin = max(0, floor(bbox[1] - 100))
    xmax = min(ceil((bbox[0] + bbox[2])), w - 1)

    # To deal with sometimes when the bbox is weird
    if bbox[3] < 1:
        bbox[3] += 100
    ymax = min(ceil((bbox[1] + bbox[3])), h - 1)

    extracted_mask = segmentation_mask[ymin:ymax, xmin:xmax]
    return np.asfortranarray(extracted_mask)


def generate_binary_mask(seg_mask: np.ndarray,
                         category_code: int or tuple) -> tuple:
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
    if isinstance(category_code, int):
        bin_mask = seg_mask == category_code
    else:
        r_bool = seg_mask[:, :, 0] == category_code[0]
        g_bool = seg_mask[:, :, 1] == category_code[1]
        b_bool = seg_mask[:, :, 2] == category_code[2]
        bin_mask = r_bool & g_bool & b_bool

    return bin_mask.astype('uint8'), np.count_nonzero(bin_mask)


def generate_oriented_annotations(instance_dir: str,
                                  xml_annotations_dir: str,
                                  categories: pd.DataFrame,
                                  img_lookup: dict,
                                  work_dir: str,
                                  train_set: set,
                                  val_set: set = None) -> tuple:
    """Generates OBB annotations only.

    Args:
        instance_dir: Path to the where the instance segmentation images are.
        xml_annotations_dir: Path to the directory containing the object
            annotations.
        categories: DataFrame containing the information in the class_names csv
            file.
        img_lookup: A lookup table to get the image_id of every named image.
        work_dir: The temporary working directory to store temp files.
        train_set: A set that includes the image names of every image in the
            training set. Used to separate which annotation list the
            annotation should be appended to.
        val_set: A set that includes the names of every image in the validation
            set. If None is given, then it is assumed every file not in the
            training set are part of the validation set.

    Returns:
        A tuple of annotation lists, the 0th element being the training ann
            dict, the 1st being the validation ann dict, the 2nd being a list of
            paths to the annotations pickled dictionary with training image ids
            as the key, and the 3rd being a list of paths to the pickled dict of
            annotations with validation image ids as the key.
    """
    print("Processing annotations...")
    train_ann_list = dict()
    val_ann_list = dict()
    train_ann_lookup = dict()
    train_ann_pickle_paths = []
    val_ann_lookup = dict()
    val_ann_pickle_paths = []
    pickle_counter = 0

    # Change index of categories to use the xml name
    cat_set = set(categories['xml_name'].to_list())
    categories.set_index('xml_name', inplace=True)

    counter = 1

    file_list = listdir(xml_annotations_dir)

    prog_bar = tqdm(file_list)

    for file_name in prog_bar:
        img_name = splitext(file_name)[0]

        img_in_train = img_name in train_set
        if val_set is not None:
            img_in_val = img_name in val_set
        else:
            img_in_val = not img_in_train

        if not (img_in_val or img_in_train):
            continue

        xml_path = join(xml_annotations_dir, file_name)
        instance_path = join(instance_dir, img_name + '_inst.png')
        file_annotations = []
        image_id = img_lookup[img_name]

        inst_array = np.array(Image.open(instance_path))
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

                # Get the instance segmentation
                extracted_seg = extract_area_inside_bbox(aligned_bbox,
                                                         inst_array)

                instance_hex = obj.find('instance').text
                instance_color = ImageColor.getrgb(instance_hex)
                bin_mask, area = generate_binary_mask(extracted_seg,
                                                      instance_color)

                if area == 0:
                    continue

                oriented_bbox = get_oriented_bbox(aligned_bbox, bin_mask)

                # Get values for comments
                comments = f"instance:{instance_hex};"
                if obj.find('onset') is not None:
                    comments += f"onset:{obj.find('onset').text.lstrip(' ')};"
                if obj.find('duration') is not None:
                    comments += f"duration:{obj.find('duration').text};"
                if obj.find('rel_position') is not None:
                    comments += f"rel_position:{obj.find('rel_position').text};"

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
                    'comments': comments
                }
                if img_in_train:
                    train_ann_list[str(counter)] = curr_ann
                else:
                    val_ann_list[str(counter)] = curr_ann
                file_annotations.append(str(counter))

                # Send train and val ann list to pickle files every million anns
                if counter % 1000000 == 0:
                    a, b, pickle_counter = pickle_anns(work_dir, pickle_counter,
                                                       train_ann_list,
                                                       val_ann_list)
                    train_ann_pickle_paths.append(a)
                    val_ann_pickle_paths.append(b)

                    train_ann_list = dict()
                    val_ann_list = dict()
                    prog_bar.write(f'Wrote pickle file {pickle_counter}')

                counter += 1
        if img_in_train:
            train_ann_lookup[image_id] = file_annotations
        elif img_in_val:
            val_ann_lookup[image_id] = file_annotations

    # Final pickling to make sure everything is pickled
    a, b, pickle_counter = pickle_anns(work_dir, pickle_counter,
                                       train_ann_list,
                                       val_ann_list)
    train_ann_pickle_paths.append(a)
    val_ann_pickle_paths.append(b)
    del train_ann_list
    del val_ann_list

    return train_ann_pickle_paths, val_ann_pickle_paths, train_ann_lookup, \
        val_ann_lookup


def pickle_anns(work_dir: str, pickle_counter: int, train_anns: dict,
                val_anns: dict) -> tuple:
    """Pickles training and validation ann dictionaries.

    Pickles the given dictionaries and returns their paths as well as the new
    pickle counter.

    Returns:
        A tuple. 0th element is fp to the train annotations pickled, 1st element
            is the fp to the val annotations pickled, 2nd element is the updated
            pickle counter.
    """
    train_ann_fp = join(work_dir,
                        f'train_anns_{pickle_counter}.pkl')

    val_ann_fp = join(work_dir,
                      f'val_anns_{pickle_counter}.pkl')
    pickle_counter += 1

    with open(train_ann_fp, 'wb') as p_file:
        pickle.dump(train_anns, p_file)
    with open(val_ann_fp, 'wb') as p_file:
        pickle.dump(val_anns, p_file)

    return train_ann_fp, val_ann_fp, pickle_counter


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


def get_aligned_bbox(bndbox: ET.Element, h: float, w: float) -> list:
    """Gets the axis aligned absolute bounding box.

    This function gets the aligned bounding box of an object based on the
    bndbox value given in the xml description of the object. h and w are used
    to change the values from relative to absolute.

    Returns:
        A list representation of the bounding box containing [xmin, xmax, w, h].
    """
    xmin = float(bndbox.find('xmin').text) * w
    ymin = float(bndbox.find('ymin').text) * h
    width = (float(bndbox.find('xmax').text) * w) - xmin
    height = (float(bndbox.find('ymax').text) * h) - ymin

    return [xmin, ymin, round(width, 8), round(height, 8)]


def get_oriented_bbox(aligned_bbox: list, bin_mask: np.ndarray) -> np.ndarray:
    """Calculates the oriented bounding box around an object.

    Returns:
        bbox as an (8,) numpy array.
    """
    adders = np.array(aligned_bbox[0:2])

    # Correction for weird misalignment issue
    adders[1] -= 4

    bin_indices = np.transpose(np.nonzero(bin_mask))
    min_box = boxPoints(minAreaRect(bin_indices))
    min_box = np.flip(min_box, 1)  # For some reason this is in the wrong order
    min_box += adders  # shift them to their actual absolute position

    return np.concatenate(min_box)
