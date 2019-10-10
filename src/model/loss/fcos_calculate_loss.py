"""Calculate Loss.

This module calculates the loss based on the given configurations.
"""
import torch

from .focal_loss import FocalLoss
from .cross_entropy_loss import CrossEntropyLoss
from .iou_loss import IoULoss
from .utils import distance2bbox
from post_processing.bbox_nms import multiclass_nms
from model.utils import multi_apply


INF = 1e8


class FCOSLossCalculator:
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
        gt_labels = self.get_batch_target_props(target, 'category_id')
        gt_bboxes = self.get_batch_target_props(target, 'bbox')
        assert len(cls_scores) == len(bbox_preds) == len(energies)

        return self._loss(cls_scores, bbox_preds, energies, gt_bboxes,
                          gt_labels)

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

    def _loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels):
        """Implementation from mmdetection"""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per image, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)