"""Bounding Box.

Bounding box class that represents abstract bounding boxes.
"""
import torch
from utils.constants import *


class BoundingBox:
    def __init__(self, bbox_output, cls_output, centerness_output, m,
                 threshold=0.99):
        """Creates a list of bounding boxes from the network output.

        Creates an abstract bounding box class from the network output. This
        class keeps track of the individual bounding boxes, and adds class
        labels to them based on the segmentation output from the network.

        One BoundingBox class should be created per head per batch.

        Shape:
            bbox_output: (4, h, w) with the first 4 being (left, top, right,
                bottom) distance from the bounding box edge.

            cls_output: (80, h, w) with the first 80 being the classes. This
                does not have to be softmaxed.

            centerness_output: (1, h, w) with the first dimension having a range
                in the interval (0, 1)

            m: (2) scaling factor on the x, y axis, respectively.

        Args:
            bbox_output (torch.Tensor): The bbox output of the network.
            cls_output (torch.Tensor): The class segmentation output of the
                network.
            centerness_output (torch.Tensor): The centerness output of the
                network.
            m (torch.Tensor): The scaling factor to scale the head output back
                to the original image size.
            threshold (float): The threshold to display a bounding box.
        """
        assert bbox_output.shape[0] == 4
        assert cls_output.shape[0] == 80
        assert centerness_output.shape[0] == 1
        assert m.shape[0] == 2

        # BBoxes are stored as a (n, 5) tensor, with n being the number of
        # bboxes and the 5 being (x0, y0, x1, y1, cat)
        cls_output = cls_output.argmax(0)
        bbox_output = self._resize_bbox(bbox_output, m)
        bbox_output = self._append_class(bbox_output, cls_output)
        bbox_output = self._suppress_non_max(bbox_output, centerness_output,
                                             threshold)
        # BBoxes are now stored in (n, 5) form, and all below the threshold has
        # been removed.
        self.bboxes = bbox_output

    def _resize_bbox(self, preds, m):
        """Turns bboxes into the standard (l,t,r,b) format and resizes them.

        Gets the bounding boxes in the standard format. Also suppresses any
        bounding boxes that are non-max. This is done simply by not doing
        operations on grid values where max_values is not 1.

        Shape:
            preds: (4, h, w) as returned by the individual head of the network.

            m: (2) scaling factor on the x, y axis respectively.

            max_values: (1, h, w) as returned by bbox_select()

        Args:
            preds (torch.Tensor): A tensor that is returned by the network and
                represents the bounding boxes. This tensor should be in
                (l, t, r, b) format, where l, t, r, and b are the left, top,
                right, and bottom distance to the edge of the bounding box.
            m (torch.Tensor): The scaling factor to scale the head output back
                to the image size.
            max_values (torch.Tensor): A byte tensor where bboxes that should be
                processed are 1 and those that should not be are 0.

        Notes:

        Bounding boxes are in the form (x0, y0, x1, y1), where x0 is the x value
        of the left edge, y0 the y value of the top edge, x1 the x value of the
        right edge, and y1 the y value of the bottom edge. This is a standard
        representation for bounding boxes.

        The network returns bounding box values for each pixel with coordinates
        (x, y) in the form (l, t, r, b), with l = x - x0, t = y - y0,
        r = x1 - x, b = y1 - y.

        To get the bounding boxes in standard form so we can display it, we
        require the (x0, y0, x1, y1) values from each pixel with:
        x0 = -l + (m * x) + s_x
        y0 = -t + (m * y) + s_y
        x1 = r + (m * x) + s_x
        y1 = b + (m * y) + s_y

        with m being the scaling factor to get the size of the head back to the
        actual size of the image and s being the shift value to move the values
        into the correct position.

        A shift value s is needed since we draw with the origin located in the
        top left, whereas the distance values are computed from the center of a
        pixel.

        The variable ONES is the required multiplier to turn (l, t, r, b) into
        (-l, -t, r, b). The variable INDEX_TENSOR contains the index values of
        each pixel, i.e. (x, y, x, y).

        Returns:
            torch.Tensor: The standardized bboxes in the shape (4, h, w).
        """
        m = m.reshape(2, 1).repeat(2, 1)

        preds *= ONES[:, :preds.shape[1], :preds.shape[2]]
        # Get m * index tensor
        a = INDEX_TENSOR[:, :preds.shape[1], :preds.shape[2]].clone()
        m_xy = a.transpose(0, 1)
        m_xy *= m
        m_xy = m_xy.transpose(0, 1)

        preds += m_xy

        # Get shift s_xy
        b = HALVES[:, :preds.shape[1], :preds.shape[2]].clone()
        s_xy = b.transpose(0, 1)
        s_xy *= m
        s_xy = s_xy.transpose(0, 1)

        preds += s_xy

        return preds

    def _suppress_non_max(self, bbox_preds, centerness, threshold):
        """Suppresses bbox values below a certain score.

        Shape:
            bbox_preds: (h, w, 5), the 5 being (x0, y0, x1, y1, cls)

            centerness: (1, h, w)

        Args:
            bbox_preds (torch.Tensor): Bounding Box predictions from the
                network.
            centerness (torch.Tensor): Centerness values from the network.
            threshold (float): Threshold value to act as cutoff.

        Returns:
            torch.Tensor: Torch tensor in the shape (n, 4) containing only
                bboxes with centerness values above the threshold where n is the
                number of bboxes.
        """
        cgt = centerness.reshape(centerness.shape[1],
                                 centerness.shape[2])
        cgt = cgt.gt(threshold)
        return bbox_preds[cgt]

    def _append_class(self, bbox_preds, cls_preds):
        """Appends class and permutes to shape (n, 5).

        Shape:
            bbox_preds: (4, h, w)
            cls_preds: (h, w)

        Args:
            bbox_preds (torch.Tensor): BBox predictions.
            cls_preds (torch.Tensor): Class predictions.

        Returns:
            torch.
        """
        cls_preds = cls_preds.unsqueeze(-1)
        bbox_preds = bbox_preds.permute(1, 2, 0)

        return torch.cat((bbox_preds, cls_preds), 2)
