"""Oriented Bounding Box Annotations

Class that represents Oriented Bounding Box annotations.

Description

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    March 03, 2020
"""
import json
from datetime import datetime
import pandas as pd
from math import isnan
import pickle


class OBBAnnotations:
    def __init__(self, description: str, subset: str, version='1.0',
                 url: str = 'https://tuggeluk.github.io/deepscores/'):
        """Creates the basic structure of the OBB annotations.

        Creates the basic structure. Images, categories, and annotations
        must be added after using the provided class methods.

        Args:
            description: Description of the DeepScores dataset.
            images: Path to the image data csv file.
        """
        self.annotations = {
            'info': {
                'description': description,
                'version': version,
                'year': int(datetime.now().strftime('%Y')),
                'contributor': 'Lukas Tuggener, Ismail Elezi, Yvan Satyawan, '
                               'Jürgen Schmidhuber, Marcello Pelillo, '
                               'Thilo Stadelmann',
                'date_created': datetime.now().strftime('%Y-%m-%d'),
                'url': url
            },
            'annotation_sets': [],
            'categories': None,
            'images': None,
            'annotations': None
        }
        self.annotations_to_include = set()

    def add_categories(self, categories: pd.DataFrame):
        """Adds categories from the given DataFrame into self."""
        categories = categories[['id', 'deepscores_category_id',
                                 'deepscores_name', 'muscima_name',
                                 'muscima_id']]
        processed_cats = {}

        for _, row in categories.iterrows():
            color, ds_id, ds_name, m_name, m_id = row
            processed_cats[str(int(ds_id))] = {
                'name': ds_name,
                'annotation_set': 'deepscores',
                'color': color
            }
            if not isnan(m_id):
                processed_cats[str(int(m_id))] = {
                    'name': m_name,
                    'annotation_set': 'muscima++',
                    'color': color
                }
        self.annotations['annotation_sets'] = [
            'deepscores', 'muscima++'
        ]

        self.annotations['categories'] = processed_cats

    def add_images(self, images: list):
        """Adds images from the given list into self."""
        self.annotations['images'] = images
        for img in images:
            self.annotations_to_include.update(img['ann_ids'])

    def add_annotations(self, annotations: list, fp_idx: int = 0):
        """Adds annotations from the given dictionary into self.

        Returns:
            int: A single value representing the index of the last file that
                was used.
        """
        dict_anns = {}
        for fp in annotations[fp_idx:]:
            with open(fp, 'rb') as pickle_file:
                curr_dict = pickle.load(pickle_file)
                keys = set(curr_dict.keys())
                if keys.issubset(self.annotations_to_include):
                    # Normal case
                    dict_anns.update(curr_dict)
                    self.annotations_to_include.difference_update(keys)
                    fp_idx += 1
                else:
                    # We're at the last file to include
                    for key in curr_dict:
                        if key in self.annotations_to_include:
                            dict_anns[key] = curr_dict[key]
                    break
        self.annotations['annotations'] = dict_anns
        return fp_idx

    def output_json(self, output_fp: str):
        """Outputs the annotations as a JSON file.

        Args:
            output_fp: Where to output the JSON file.
        """
        cat = img = ann = True
        if self.annotations['categories'] is None:
            cat = False
        if self.annotations['images'] is None:
            img = False
        if self.annotations['annotations'] is None:
            ann = False

        if not (cat and img and ann):
            print('OBB annotations has not been properly initialized. The '
                  'following required attributes are missing:')
            if not cat:
                print('- categories')
            if not img:
                print('- images')
            if not ann:
                print('- annotations')
            return -1

        print('Writing to disk...')
        with open(output_fp, 'w+') as output_file:
            json.dump(self.annotations, output_file)
