"""DSaM MUSCIMA ID.

Giving the MUSCIMA classes an appropriate ID is error prone when making sure
that each ID is unique, but the same class always has the same ID. So this
script does that automatically.
"""
from csv import DictReader, DictWriter
from os.path import splitext
from argparse import ArgumentParser


def parse_args():
    """Parses arguments."""
    p = ArgumentParser(description='gives appropriate IDs to the MUSCIMA '
                                   'classes')
    p.add_argument('PATH', type=str, help='path to the file to be processed')
    return p.parse_args()


def main(fp):
    lines = []
    counter = 1
    max_ds_id = 0
    id_mappings = {}
    with open(fp) as f:
        reader = DictReader(f)
        for line in reader:
            muscima_name = line['muscima_name']
            if muscima_name is not '':
                if muscima_name not in id_mappings:
                    # If we haven't encountered the cat yet, add it
                    id_mappings[muscima_name] = counter
                    counter += 1
            lines.append(line)
            if int(line['deepscores_category_id']) > max_ds_id:
                # Used to fine the max id from deepscores
                max_ds_id = int(line['deepscores_category_id'])

    for muscima_cat in id_mappings:
        # Make sure ids are unique with deepscores cat ids
        id_mappings[muscima_cat] += max_ds_id

    for line in lines:
        if line['muscima_name'] is not '':
            line['muscima_id'] = id_mappings[line['muscima_name']]

    header = list(lines[0].keys())

    out_file = splitext(fp)[0] + '_processed' + splitext(fp)[1]
    with open(out_file, 'w', newline='') as f:
        writer = DictWriter(f, header)
        writer.writeheader()
        writer.writerows(lines)


if __name__ == '__main__':
    main(parse_args().PATH)