"""COCO Annotations.

Class that holds the COCO-like annotations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 18, 2019
"""
import json
from csv import DictReader


class CocoLikeAnnotations:
    def __init__(self, description: str, images: str, categories: list,
                 annotations: list,
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
        print('Reading image CSV file...')
        image_list = []
        with open(self.annotations['images'], mode='r') as csvfile:
            reader = DictReader(csvfile, skipinitialspace=True)
            for row in reader:
                image_list.append({
                    'license': int(row['license']),
                    'file_name': row['file_name'],
                    'coco_url': row['coco_url'],
                    'height': int(row['height']),
                    'width': int(row['width']),
                    'date_captured': row['date_captured'],
                    'flickr_url': row['flickr_url'],
                    'id': int(row['id'])
                })

        annotation = {
            'info': self.annotations['info'],
            'licenses': self.annotations['licenses'],
            'images': image_list,
            'annotations': self.annotations['annotations'],
            'categories': self.annotations['categories']
        }
        print('Writing to disk...')
        with open(output_fp, 'w+') as output_file:
            json.dump(annotation, output_file#, indent=4,
                      # separators=(',', ': ')
                      )
