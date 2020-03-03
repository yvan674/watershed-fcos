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
                'contributor': 'Lukas Tuggener, Ismail Elezi, Jürgen '
                               'Schmidhuber, Marcello Pelillo, '
                               'Thilo Stadelmann, Yvan Satyawan',
                'date_created': datetime.now().strftime('%Y-%m-%d'),
            },
            'categories': None,
            'images': None,
            'annotations': None
        }

    def add_categories(self, categories: dict):
        """Adds categories from the given dictionary into self."""
        self.annotations['categories'] = categories

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
        print('Writing to disk...')
        with open(output_fp, 'w+') as output_file:
            json.dump(self.annotations, output_file)
