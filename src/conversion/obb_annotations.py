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


class OBBAnnotations:
    def __init__(self, description: str, version='1.0',
                 url: str='https://tuggeluk.github.io/deepscores/'):
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
                'contributor': 'Lukas Tuggener, Ismail Elezi,  Yvan Satyawan, '
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

    def add_categories(self, categories: pd.DataFrame):
        """Adds categories from the given DataFrame into self."""
        if ann_set_name not in self.annotations['annotation_sets']:
            self.annotations['annotation_sets'].append(ann_set_name)

        categories = categories[['id', 'deepscores_category_id',
                                 'deepscores_name', 'muscima_name',
                                 'muscima_id']]
        processed_cats = {}

        for _, row in categories.iterrows():
            color, ds_id, ds_name, m_name, m_id = row
            processed_cats[str(ds_id)] = {
                'name': ds_name,
                'annotation_set': 'deepscores',
                'color': color
            }
            processed_cats[str(m_id)] = {
                'name': m_name,
                'annotation_set': 'muscima++',
                'color': color
            }

        self.annotations['categories'] = processed_cats

    def add_images(self, images: list):
        """Adds images from the given list into self."""
        self.annotations['images'] = images

    def add_annotations(self, annotations: dict):
        """Adds annotations from the given dictionary into self."""
        self.annotations['annotations'] = annotations

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
