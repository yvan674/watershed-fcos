"""COCO Annotations.

Class that holds the COCO-like annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 18, 2019
"""
import json


class CocoLikeAnnotations:
    def __init__(self, description: str, images: list, categories: list,
                 annotations: list):
        """Creates the basic structure of the COCO-like dataset.

        Args:
            description: Description of the DeepScores dataset.
            images: List of the images according to the COCO conventions.
            categories: List of the categories according to the COCO
                conventions.
            annotations: List of the annotations according to the COCO
                conventions.
        """
        self.annotations = {
            'info': {
                'description': description,
                'url': 'https://tuggeluk.github.io/deepscores/',
                'version': '1.0'
            },

            'licenses': [
                {
                    'url': 'https://opensource.org/licenses/MIT',
                    'id': 1,
                    'name': 'MIT License'
                }
            ],
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

    def output_json(self, output_fp: str):
        """Outputs the annotations file as a JSON file.

        Args:
            output_fp: Where to output the JSON file.
        """
        with open(output_fp, 'w+') as output_file:
            json.dump(self.annotations, output_file, indent=4,
                      separators=(',', ': '))
