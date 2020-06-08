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
import pandas as pd
from re import finditer
import json


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                       identifier)
    return [m.group(0) for m in matches]


def csv_bool_to_python_bool(string: str) -> bool:
    """Converts a CSV bool to a python bool object.

    CSV saves boolean data as 'TRUE' or 'FALSE'. Python thinks these are both
    true. This function converts these CSV strings to python bool objects.

    Args:
        string: The CSV string to be converted.

    Returns:
        The python version of the string.
    """
    assert string in ('TRUE', 'FALSE')
    return string == 'TRUE'


def generate_categories(file_path: str) -> tuple:
    """Generates a COCO categories list and a lookup table from class_names.csv

    Args:
        file_path: path to the class_names.csv file

    Returns:
         A tuple where the first item is a COCO-like list of items. The second
         is a lookup table dictionary with the key being the file name and the
         value being the id number. The third is a set of all category names.
    """
    categories = []
    lookup_table = dict()
    categories_set = set()
    with open(file_path) as class_names:
        temp_reader = csv.reader(class_names)
        contains_blacklist = len(next(temp_reader)) == 3
        del temp_reader
        reader = csv.reader(class_names)
        if contains_blacklist:
            for id, name, blacklist in reader:
                if csv_bool_to_python_bool(blacklist):
                    # Don't process if blacklist
                    continue

                split_name = camel_case_split(name)
                super_category = split_name[0]
                if super_category == "notehead":
                    categories.append({
                        'supercategory': super_category,
                        'id': str(int(id) + 1000),
                        'name': name + 'Online'
                    })
                    lookup_table[name + 'Online'] = str(int(id) + 1000)
                    categories_set.add(name + 'Online')
                    categories.append({
                        'supercategory': super_category,
                        'id': id,
                        'name': name + 'Offline'
                    })
                else:
                    categories.append({
                        'supercategory': super_category,
                        'id': id,
                        'name': name
                    })
                lookup_table[name] = id
                categories_set.add(name)
        else:
            for id, name in reader:
                split_name = camel_case_split(name)
                super_category = split_name[0]

                categories.append({
                    'supercategory': super_category,
                    'id': id,
                    'name': name
                })
                lookup_table[name] = id
                categories_set.add(name)

    return categories, lookup_table, categories_set


def generate_oriented_categories(fp: str) -> pd.DataFrame:
    """Generates categories according to the OBB schema.

    Does data validation and converts category ids to ints.
    Args:
        fp: path to the class_names.csv file

    Returns:
        DataFrame of the categories csv file.
    """
    cats = pd.read_csv(fp)
    cats['deepscores_category_id'] = cats['deepscores_category_id'].map(
        lambda x: int(x)
    )
    cats['muscima_id'] = cats['muscima_id'].map(lambda x: int(x))
    return pd.read_csv(fp)


if __name__ == '__main__':
    categories, lookup_table = generate_categories(
        '/Users/Yvan/Offline Files/deep_scores_dense/class_names.csv/'
    )

    print(json.dumps(categories, indent=4, separators=(',', ': ')))
