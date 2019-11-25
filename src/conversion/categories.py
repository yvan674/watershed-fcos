"""Categories.

Categories as defined by file names in the meta_info directory.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    camel_case_splitter found from stackoverflow at the following link:
    <https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-
        in-python>
"""
import csv
from re import finditer
import json


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                       identifier)
    return [m.group(0) for m in matches]


def generate_categories(file_path: str) -> tuple:
    """Generates a COCO categories list and a lookup table from class_names.csv

    Args:
        file_path: path to the class_names.csv file

    Returns:
         A tuple where the first item is a COCO-like list of items. The second
         is a lookup table dictionary with the key being the file name and the
         value being the id number.
    """
    categories = []
    lookup_table = dict()
    with open(file_path) as class_names:
        reader = csv.reader(class_names)
        for id, name in reader:
            split_name = camel_case_split(name)
            super_category = split_name[0]
            categories.append({
                'supercategory': super_category,
                'id': id,
                'name': name
            })
            lookup_table[name] = id

    return categories, lookup_table


if __name__ == '__main__':
    categories, lookup_table = generate_categories(
        '/Users/Yvan/Offline Files/deep_scores_dense/class_names.csv/'
    )

    print(json.dumps(categories, indent=4, separators=(',', ': ')))

