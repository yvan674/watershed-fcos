from PIL import ImageDraw
from bboxes.classes_lookup import *
from utils.colormapping import map_color_value


def draw_boxes(image, bounding_boxes):
    """Draw all bounding_boxes on the image along with their class.

    Args:
        image (PIL.Image.Image): Image to draw the bounding boxes on
        bounding_boxes (list): List of BoundingBox objects. Each BoundingBox
            object represents the bounding box outputs of a head.

    Returns:
        PIL.Image.Image: Image with bounding boxes drawn on top.
    """
    draw = ImageDraw.Draw(image)

    for bbs in bounding_boxes:
        for bb in bbs.bboxes:
            bb = bb.numpy()
            # Map category to color
            color = map_color_value(bb[4], 80)

            # Get category class label
            label = CATS[bb[4]]["name"]

            # Draw the bounding box
            draw.rectangle(bb[0:4], outline=color)

            # Draw the category label
            draw.text(bb[0:2], label, fill=(255, 255, 255))
            draw.rectangle([bb[0], bb[1], bb[0] + 50, bb[1] + 20])
    return image
