"""BBox Processing.

Processes bounding boxes for use by the tensorboard SummaryWriter
"""
from .constants import *
import torch


def bbox_select(ec_map, threshold=0.99):
    """Suppresses bounding boxes which do not correspond to a high centerness.

    Each cell in the feature map output returns a bounding box. This function
    returns a list of the indexes of of the grid values where the cell of the
    energy or centerness map is above the threshold value.

    Shape:
        ec_map: Each element is in the shape (batch, 1, h, w).

        output: A list, with each element being a list with the shape (x, y).

    Args:
        ec_map (list): A list of torch.Tensor objects. Each element of this list
            represents the output of the different heads of the network. ec_map
            stands for energy centerness map.
        threshold (float): The threshold value for which bboxes to show. This
            threshold values means the the top nth percent of bboxes will be
            shown.

    Returns:
        list: A list of torch.Tensor objects. Each element of this list
            represents the output of the different heads of the network. Each
            tensor object is a byte tensor where 1 represents
    """
    out = []
    flattened_heads = []
    batch_thresholds = []
    for _ in range(ec_map[0].shape[0]):
        # Create empty lists for each batch
        flattened_heads.append([])

    for head in ec_map:
        for batch in enumerate(head):
            hxw = batch[1].shape[1] * batch[1].shape[2]
            flattened_heads[batch[0]].append(batch[1].reshape(hxw))

    for batch in range(len(flattened_heads)):
        flattened_heads[batch] = torch.cat(flattened_heads[batch])
        k = int(threshold * flattened_heads[batch].shape[0])
        batch_thresholds.append(flattened_heads[batch].kthvalue(k))
    # Now we have a kth percentile value for each batch stored in
    # batch_thresholds

    for head in ec_map:
        temp_head = []
        for batch in enumerate(head):
            temp_head.append(torch.gt(batch[1],
                                      batch_thresholds[batch[0]].values)
                             .unsqueeze(0))
        temp_head = torch.cat(temp_head)
        out.append(temp_head)

    return out


def resize_bboxes(preds, m, max_values):
    """Gets bounding boxes in standard (l,t,r,b) format and suppresses non-max.

    Gets the bounding boxes in the standard format. Also suppresses any
    bounding boxes that are non-max. This is done simply by not doing operations
    on grid values where max_values is not 1.

    Shape:
        preds: (4, h, w) as returned by the individual head of the network.

        m: (2) scaling factor on the x, y axis respectively.

        max_values: (1, h, w) as returned by bbox_select()

    Args:
        preds (torch.Tensor): A tensor that is returned by the network and
            represents the bounding boxes. This tensor should be in (l, t, r, b)
            format, where l, t, r, and b are the left, top, right, and bottom
            distance to the edge of the bounding box.
        m (torch.Tensor): The scaling factor to scale the head output back to
            the image size.
        max_values (torch.Tensor): A byte tensor where bboxes that should be
            processed are 1 and those that should not be are 0.

    Notes:

    Bounding boxes are in the form (x0, y0, x1, y1), where x0 is the x value of
    the left edge, y0 the y value of the top edge, x1 the x value of the right
    edge, and y1 the y value of the bottom edge. This is a standard
    representation for bounding boxes.

    The network returns bounding box values for each pixel with coordinates
    (x, y) in the form (l, t, r, b), with l = x - x0, t = y - y0, r = x1 - x,
    b = y1 - y.

    To get the bounding boxes in standard form so we can display it, we require
    the (x0, y0, x1, y1) values from each pixel with:
    x0 = -l + (m * x) + s_x
    y0 = -t + (m * y) + s_y
    x1 = r + (m * x) + s_x
    y1 = b + (m * y) + s_y

    with m being the scaling factor to get the size of the head back to the
    actual size of the image and s being the shift value to move the values into
    the correct position.

    A shift value s is needed since we draw with the origin located in the top
    left, whereas the distance values are computed from the center of a pixel.

    The variable ONES is the required multiplier to turn (l, t, r, b) into
    (-l, -t, r, b). The variable INDEX_TENSOR contains the index values of each
    pixel, i.e. (x, y, x, y).

    Returns:
        torch.Tensor: The standardized bboxes in the shape (4, h * w), i.e. a
            list of all the bboxes from each pixel coordinate.
    """
    assert m.shape == torch.Size([2])

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

    preds *= max_values.to(dtype=preds.dtype)

    # Reshape bboxes to [4, h * w], i.e. a list of all the bboxes
    return preds.reshape(4, preds.shape[1] * preds.shape[2])
