"""Calculate Loss.

This module calculates the loss based on the given configurations.
"""
from .focal_loss import FocalLoss
from .cross_entropy_loss import CrossEntropyLoss
from .iou_loss import IoULoss


class LossCalculator:
    def __init__(self, classification_loss_cfg, bbox_loss_cfg, energy_loss_cfg):
        """Calculates loss of the model.

        Args:
            classification_loss_cfg (dict): Configuration for classification
                loss.
            bbox_loss_cfg (dict): Configuration for bbox loss.
            energy_loss_cfg (dict): Configuration for energy loss.
        """
        self.loss_cls = FocalLoss(**classification_loss_cfg)
        self.loss_bbox = IoULoss(**bbox_loss_cfg)
        self.loss_energy = CrossEntropyLoss(**energy_loss_cfg)

    def calculate(self, input, target):
        """Calculate losses for a given input-target pair"""
        raise NotImplementedError

    def backwards(self):
        """Calls each loss' backwards() function"""
        self.loss_cls.backwards()
        self.loss_bbox.backwards()
        self.loss_energy.backwards()