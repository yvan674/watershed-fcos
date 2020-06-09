"""Dataset Statistics.

Calculates the statistics of the datasets. Specificially, calculates "white"
area of each bounding box, as well as overlapping pixels per bounding box.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    April 8, 2020
"""
from argparse import ArgumentParser
from os.path import join, exists
from os import mkdir
from time import time
import json
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from shapely.geometry import Polygon
from conversion.deepscores_conversion_obb import pretty_time_delta


def parse_args():
    """Parses arguments."""
    p = ArgumentParser(description='calculates statistics for a coco-style '
                                   'dataset')
    p.add_argument('TRAIN', type=str,
                   help='path of the training annotation file to process')
    p.add_argument('VAL', type=str,
                   help='path of the validation annotation file to process')
    p.add_argument('OUT', type=str,
                   help='path of the dir to write the csv file out to')
    p.add_argument('-x', action='store_true',
                   help='only calculates whitespace')
    p.add_argument('-n', type=int, nargs='?',
                   help='number of workers to use')
    return p.parse_args()


def process_ann_row(row):
    """Get aligned bbox of a row."""
    o_bbox = row['o_bbox']
    a_bbox = row['a_bbox']
    xs = np.array(o_bbox[::2])
    ys = np.array(o_bbox[1::2])
    o_area = float(0.5 * np.abs(np.dot(xs, np.roll(ys, 1))
                                - np.dot(ys, np.roll(xs, 1))))

    a_area = float(abs(a_bbox[0] - a_bbox[2]) * abs(a_bbox[1] - a_bbox[3]))
    if row['area'] < 0:
        a_ws = 0.
        o_ws = 0.
    else:
        if a_area > 0.:
            a_ws = 1 - max(0., (float(row['area'] / a_area)))

        else:
            a_ws = 0.
        if o_area > 0.:
            o_ws = max(0., 1 - (float(row['area'] / o_area)))
        else:
            o_ws = 0.

    return pd.Series({'aligned_whitespace': float(a_ws),
                      'oriented_whitespace': float(o_ws),
                      'area': row['area'],
                      'a_bbox0': float(a_bbox[0]),
                      'a_bbox1': float(a_bbox[1]),
                      'a_bbox2': float(a_bbox[2]),
                      'a_bbox3': float(a_bbox[3]),
                      'o_bbox_x0': float(xs[0]),
                      'o_bbox_x1': float(xs[1]),
                      'o_bbox_x2': float(xs[2]),
                      'o_bbox_x3': float(xs[3]),
                      'o_bbox_y0': float(ys[0]),
                      'o_bbox_y1': float(ys[1]),
                      'o_bbox_y2': float(ys[2]),
                      'o_bbox_y3': float(ys[3]),
                      'cat_id': str(row['cat_id'][0]),
                      'img_id': row['img_id']})


def calc_single_overlap(row, index, a_bbox0v, a_bbox1v, a_bbox2v, a_bbox3v,
                  o_bboxxv, o_bboxyv):

    if row.name == index:
        # Don't calc against itself.
        return pd.Series({'aligned': False, 'oriented': False})

    # a_ol stands for aligned_overlap
    a_ol = abs(row['a_bbox0'] - a_bbox0v) * 2 < (row['a_bbox2'] + a_bbox2v)
    if a_ol:
        a_ol &= abs(row['a_bbox1'] - a_bbox1v) * 2 < (row['a_bbox3'] + a_bbox3v)

    # Only check for oriented intersection if aligned has overlap
    if a_ol:
        # Create Polygon from row and value oriented bboxes
        pr = Polygon(
            [(x, y) for x, y in zip(row['o_bboxx'], row['o_bboxy'])])
        pv = Polygon(
            [(x, y) for x, y in zip(o_bboxxv, o_bboxyv)])

        o_ol = pv.intersects(pr)
    else:
        o_ol = False

    return pd.Series({'aligned': a_ol, 'overlap': o_ol})


def main(train_file, val_file, out_dir, whitespace_only, num_workers=None):
    """Main loop.

    Calculates whitespace in each bbox and overlaps

    :param str train_file: Path to the training annotation file.
    :param str val_file: Path to the validation annotation file.
    :param str out_dir: Where to output the results to.
    :param bool whitespace_only: Only calculate whitespace results.
    :param int num_workers: Number of workers to initialize for pandarallel
    """
    if not whitespace_only:
        raise NotImplementedError('Was too time consuming to implement.')
    if num_workers is None:
        pandarallel.initialize()
    else:
        pandarallel.initialize(nb_workers=num_workers, progress_bar=False)

    if not exists(out_dir):
        mkdir(out_dir)

    start_time = time()
    print("Loading training annotation file...")
    with open(train_file) as fp:
        train_file = json.load(fp)

    anns = pd.DataFrame.from_dict(train_file['annotations'], orient='index')
    cats = train_file['categories']

    print("Loading validation annotation file...")
    with open(val_file) as fp:
        val_file = json.load(fp)
    val_anns = pd.DataFrame.from_dict(val_file['annotations'], orient='index')
    anns = pd.concat([anns, val_anns])
    del val_anns
    # ####################### DEBUG #######################
    # print("Loading validation annotation file...")
    # with open(val_file) as fp:
    #     val_file = json.load(fp)
    # anns = pd.DataFrame.from_dict(val_file['annotations'], orient='index')
    # anns = anns.iloc[:500]
    # cats = val_file['categories']
    # # ####################### DEBUG #######################

    # Do cleanup to reduce memory footprint
    del train_file
    del val_file

    # Prune to only DeepScores classes
    cats = {k: v
            for k, v in cats.items()
            if v['annotation_set'] == 'deepscores'}

    print(f"Done! t={time() - start_time:.2f}s")

    print("Processing annotations...")
    start_time = time()
    anns = anns.parallel_apply(process_ann_row, axis=1)
    anns = anns.loc[anns['area'] > 0]  # Keep only annotations with area > 0

    # if not whitespace_only:
    #     overlaps = anns.parallel_apply(calc_overlaps, axis=1, args=anns)
    print(f"Done! t={pretty_time_delta(time() - start_time)}")

    print('Doing post processing...')
    whitespace = anns[['aligned_whitespace', 'oriented_whitespace', 'cat_id']]
    whitespace = whitespace.groupby('cat_id', sort=True)
    cat_counts = whitespace.size()
    whitespace = whitespace.sum()

    mean_whitespace = whitespace.mean()
    mean_whitespace /= cat_counts.sum()
    whitespace['aligned_whitespace'] /= cat_counts
    whitespace['oriented_whitespace'] /= cat_counts


    # if not whitespace_only:
    #     mean_overlap = overlap_df.mean()
    #     overlap_df['aligned'] = overlap_df['aligned'].div(cat_counts)
    #     overlap_df['oriented'] = overlap_df['oriented'].div(cat_counts)

    # Prepare for csv file. Done manually because I'm too lazy to verify
    # which join type I should use and how to properly rename columns tbh and
    # this was written before moving to pandas as the data structure
    header = 'cat_id,aligned_whitespace,oriented_whitespace{}\n'.format(
        ',aligned_overlap,oriented_overlap' if not whitespace_only else ''
    )
    lines = [header]


    for cat_id, row in whitespace.iterrows():
        cat_name = cats[str(cat_id)]['name']
        data_line = f'{cat_name},' \
                    f'{row["aligned_whitespace"]},' \
                    f'{row["oriented_whitespace"]}'

        # if not whitespace_only:
        #     cat_overlap = overlap_df.loc[k]
        #     data_line += f',{cat_overlap["aligned"]}' \
        #                  f',{cat_overlap["oriented"]}'

        lines.append(data_line + "\n")

    data_line = f'mean,{mean_whitespace["aligned_whitespace"]},' \
                f'{mean_whitespace["oriented_whitespace"]}'

    # if not whitespace_only:
    #     data_line += f',{mean_overlap["aligned"]},{mean_overlap["oriented"]}'

    lines.append(data_line + '\n')

    print('Writing out file...')
    with open(join(out_dir, 'results.csv'), mode='w') as fp:
        fp.writelines(lines)

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main(args.TRAIN, args.VAL, args.OUT, args.x, args.n)
