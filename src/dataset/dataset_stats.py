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
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
from shapely.geometry import Polygon


def parse_args():
    """Parses arguments."""
    p = ArgumentParser(description='calculates statistics for a coco-style '
                                   'dataset')
    p.add_argument('ANN', type=str,
                   help='paths of the annotation file to process')
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
    o_area = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1))
                          - np.dot(ys, np.roll(xs, 1)))
    a_area = a_bbox[2] * a_bbox[3]

    result = [a_bbox[0], a_bbox[1], a_bbox[2], a_bbox[3],
              a_area,
              xs, ys, o_area,
              row['area'], row['cat_id']]
    indices = ['a_bbox0', 'a_bbox1', 'a_bbox2', 'a_bbox3', 'a_area',
               'o_bboxx', 'o_bboxy', 'o_area', 'area', 'cat_id']

    return pd.Series(result, index=indices)


def calc_whitespace(x):
    if x['a_area'] == 0:
        a_ws = 0.
    else:
        a_ws = 1 - (x['area'] / x['a_area'])

    if x['o_area'] == 0:
        o_ws = 0.
    else:
        o_ws = 1 - (x['area'] / x['o_area'])

    return pd.Series([a_ws, o_ws], index=['a_ws', 'o_ws'])


def calc_overlaps(row, index, a_bbox0v, a_bbox1v, a_bbox2v, a_bbox3v,
                  o_bboxxv, o_bboxyv):
    indices = ['a_ol', 'o_ol']

    if row.name == index:
        # Don't calc against itself.
        return pd.Series([False, False], index=indices)

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

    results = [a_ol, o_ol]

    return pd.Series(results, index=indices)


def main(ann_file, out_dir, whitespace_only, num_workers=None):
    """Main loop.

    Calculates whitespace in each bbox and overlaps

    :param str ann_file: Path to the annotation file to get statistics for.
    :param str out_dir: Where to output the results to.
    :param bool whitespace_only: Only calculate whitespace results.
    :param int num_workers: Number of workers to initialize for pandarallel
    """
    if not whitespace_only:
        if num_workers is None:
            pandarallel.initialize()
        else:
            pandarallel.initialize(nb_workers=num_workers)

    if not exists(out_dir):
        mkdir(out_dir)

    print("Loading annotation file...")
    with open(ann_file) as fp:
        file = json.load(fp)

    imgs = file['images']
    anns = pd.DataFrame.from_dict(file['annotations'], orient='index')
    cats = file['categories']

    # Aligned statistics will have the prefix a_
    # OBB statistics will have the prefix o_
    # Whitespace stats will have the suffix _ws
    # Overlap stats will have the suffix _ol
    a_ws, o_ws, a_ol, o_ol, cat_counts = {}, {}, {}, {}, {}
    for k in cats.keys():
        a_ws[k] = 0.
        o_ws[k] = 0.
        a_ol[k] = 0.
        o_ol[k] = 0.
        cat_counts[k] = 0

    total_anns = 0

    for img in tqdm(imgs[:20], unit='imgs'):
        ann_ids = [str(i) for i in img['ann_ids']]
        img_anns = anns.loc[ann_ids]
        processed = img_anns.apply(process_ann_row, axis=1)
        processed = processed.loc[processed['area'] > 0]
        total_anns += len(processed)

        # Now we have a new dataframe with all the info we need.
        for k in cats.keys():
            this_cat = processed.loc[processed['cat_id'] == k]
            if len(this_cat) == 0:
                continue
            # Process whitespace first
            ws = this_cat.apply(calc_whitespace, axis=1)
            ws = ws.apply(np.sum)
            ws = ws.apply(np.clip, args=(0, None))
            a_ws[k] += ws['a_ws']
            o_ws[k] += ws['o_ws']
            cat_counts[k] += len(this_cat)

        # Then process overlaps
        if not whitespace_only:
            for ann in processed.itertuples():
                overlaps = processed.parallel_apply(
                    calc_overlaps,
                    axis=1,
                    args=(ann.Index, ann.a_bbox0, ann.a_bbox1, ann.a_bbox2,
                          ann.a_bbox3, ann.o_bboxx, ann.o_bboxy)
                )
                overlaps = overlaps.apply(np.sum)
                c_id = str(ann.cat_id)
                a_ol[c_id] += overlaps['a_ol']
                o_ol[c_id] += overlaps['o_ol']

    print('Doing final post_processing...')

    mean_a_ws = 0
    mean_a_ol = 0
    mean_o_ws = 0
    mean_o_ol = 0

    # Prepare for csv file
    lines = ['cat_id,aligned_whitespace,oriented_whitespace,aligned_overlap,'
             'oriented_overlap\n']
    for k in cats.keys():
        if cat_counts[k] > 0:
            mean_a_ws += a_ws[k]
            mean_a_ol += a_ol[k]
            mean_o_ws += o_ws[k]
            mean_o_ol += o_ol[k]

            a_ws[k] /= cat_counts[k]
            o_ws[k] /= cat_counts[k]
            a_ol[k] /= cat_counts[k]
            o_ol[k] /= cat_counts[k]
            cat_name = cats[k]

            lines.append(f'{cat_name},{a_ws[k]},{o_ws[k]},{a_ol[k]},'
                         f'{o_ol[k]}\n')

    mean_a_ws /= total_anns
    mean_a_ol /= total_anns
    mean_o_ws /= total_anns
    mean_o_ol /= total_anns

    lines.append(f'mean,{mean_a_ws},{mean_o_ws},{mean_a_ol},{mean_o_ol}\n')

    print('Writing out file...')
    with open(join(out_dir, 'results.csv'), mode='w') as fp:
        fp.writelines(lines)

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main(args.ANN, args.OUT, args.x, args.n)