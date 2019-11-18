"""Meta Analyze.

Analyzes the meta_info files.
"""
from csv import DictReader
import os
import sys


def check_for_positive_unnamed(file_list: list) -> int:
    """Checks if any file has an entry with an unnamed value of > 0.

    Args:
        file_list: List of file paths that should be checked.
    """
    greater_than_zero_count = 0
    for file_name in file_list:
        with open(file_name, mode='r') as file:
            reader = DictReader(file)
            for row in reader:
                if float(row["Unnamed: 0"]) > 0:
                    greater_than_zero_count += 1

    return greater_than_zero_count


if __name__ == '__main__':
    try:
        file_dir = sys.argv[1]
    except IndexError:
        sys.exit(1)
    file_list = os.listdir(file_dir)
    absolute_file_list = []
    for file in file_list:
        absolute_file_list.append(os.path.join(file_dir, file))

    print(check_for_positive_unnamed(absolute_file_list))
