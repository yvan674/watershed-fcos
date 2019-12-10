"""Mean and Standard Deviation.

Calculates the mean and Standard Deviation of a dataset.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    December 02, 2019
"""
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


def parse_arguments():
    """Parses cmd line arguments."""
    parser = argparse.ArgumentParser(description='Calculates mean and standard '
                                                 'deviation for a dataset.')

    parser.add_argument('DIR', nargs=1,
                        help='where the images are.')

    return parser.parse_args()


class ImageOnlyDataset(Dataset):
    def __init__(self, image_directory_path):
        super(ImageOnlyDataset, self).__init__()
        self.data = []
        for dp, _, fn in os.walk(image_directory_path):
            for f in fn:
                if '.png' in f:
                    self.data.append(os.path.join(dp, f))

    def __getitem__(self, idx):
        return np.array(Image.open(self.data[idx]))

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    arguments = parse_arguments()

    ds = ImageOnlyDataset(arguments.DIR[0])
    dl = DataLoader(ds, 1)
    mean = [0, 0, 0]
    std = [0, 0, 0]
    print('Processing images...')

    for n, data in enumerate(tqdm(dl)):
        for i in range(3):
            mean[i] += data.to(dtype=torch.float)[:,:,:,i].mean().item()
            std[i] += data.to(dtype=torch.float)[:,:,:,i].std().item()

        print('Processed {} of {} images.'.format(n, len(ds)))

    for i in range(3):
        mean[i] /= len(ds)
        std[i] /= len(ds)

    print('mean:')
    print(mean)
    print('\nstd:')
    print(std)