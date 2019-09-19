"""Coco Dataset.

Coco Dataset without any images with crowds. Done temporarily to prevent issues
with the transformation.
"""
from torchvision.datasets import CocoDetection


class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CocoDataset, self).__init__(root, annFile, transforms=transforms)
        self.length = len(self)

    def __getitem__(self, index: int):
        is_crowd = True
        while is_crowd:
            is_crowd = False
            img, annotations = super().__getitem__(index)
            for annotation in annotations:
                if annotation['iscrowd'] == 1:
                    is_crowd = True
                    index += 1
                    if index >= self.length:
                        index = 0
                    break
        return img, annotations
