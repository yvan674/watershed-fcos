"""Calculate Loss.

This module calculates the loss based on the given configurations.
"""
from .focal_loss import FocalLoss
from .cross_entropy_loss import CrossEntropyLoss
from .iou_loss import IoULoss


class WFCOSLossCalculator:
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

    def calculate(self, prediction, target):
        """Calculate losses for a given input-target pair.

        Args:
            prediction (tuple): Prediction from the network output as a tuple of
                tensors.
            target (list): List of dicts which are the targets as defined by the
                COCO annotations file.

        Notes:
            Each of the target objects are represented as a dict with keys:
            ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox',
            'category_id', 'id']
        """
        # Prep for easier accesses/clearer naming and assert it's correctly
        # sized
        cls_scores = [preds[1][0] for preds in prediction]
        bbox_preds = [preds[1][1] for preds in prediction]
        energies = [preds[1][2] for preds in prediction]
        labels = self.get_batch_target_props(target, 'category_id')
        gt_bboxes = self.get_batch_target_props(target, 'bbox')
        assert len(cls_scores) == len(bbox_preds) == len(energies)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels_bbox_Targets = self.fcos_target(all_level_points, )


        loss_cls = self.loss_cls(prediction, target[''])

    def get_batch_target_props(self, targets, property):
        """Gets a property from target annotations of an entire batch.

        Args:
            targets (list): List of targets for the batch.
            property (str): The annotation property to get.
        """
        batch = []
        for target in targets:
            objects = []
            for object in target:
                objects.append(object[property])
            batch.append(objects)
        return batch

    def backwards(self):
        """Calls each loss' backwards() function"""
        self.loss_cls.backwards()
        self.loss_bbox.backwards()
        self.loss_energy.backwards()