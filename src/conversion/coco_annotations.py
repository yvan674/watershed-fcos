"""COCO Annotations.

Class that holds the COCO-like annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 18, 2019
"""
import json


class CocoLikeAnnotations:
    def __init__(self, description: str, version='1.0',
                 url: str='https://tuggeluk.github.io/deepscores/'):
        """Creates the basic structure of the COCO-like dataset.

        Args:
            description: Description of the DeepScores dataset.
            images: Path to the image data csv file.
            categories: List of the categories according to the COCO
                conventions.
            annotations: Dictionary containing the path and shape of the
                memory-mapped annotation arrays.
        """
        self.annotations = {
            'info': {
                'description': description,
                'url': url,
                'version': version
            },

            'licenses': [
                {
                    'url': 'https://opensource.org/licenses/MIT',
                    'id': 1,
                    'name': 'MIT License'
                }
            ],
            'images': None,
            'annotations': None,
            'categories': None
        }

    def add_categories(self, categories: dict):
        self.annotations['categories'] = categories

    def add_images(self, images: list):
        self.annotations['images'] = images

    def add_annotations(self, annotations: dict):
        self.annotations['annotations'] = annotations

    def output_json(self, output_fp: str):
        """Outputs the annotations file as a JSON file.

        Args:
            output_fp: Where to output the JSON file.
        """
        print('Writing to disk...')
        with open(output_fp, 'w+') as output_file:
            json.dump(self.annotations, output_file#, indent=4,
                      # separators=(',', ': ')
                      )
