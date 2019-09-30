"""Constants.

Constants are kept here for easy modifications whenever needed.
"""
import torch


NUM_CLASSES = 80
IMAGE_SIZE = (640, 800)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    CUDA_COUNT = torch.cuda.device_count()
else:
    DEVICE = torch.device("cpu")
    CUDA_COUNT = 0

# Create index tensors. First, create an arange preds from 0 to x and y
# respectively. Then, for y specifically, change the view so it's a column
# preds instead of a row preds. Repeat both tensors by y and x respectively.
# Finally, unsqueeze to give it a 0th dimension
x_index = torch.arange(IMAGE_SIZE[1]).repeat(IMAGE_SIZE[0], 1).unsqueeze(0)
y_index = torch.arange(IMAGE_SIZE[0]).view(-1, 1).repeat(1, IMAGE_SIZE[1])\
    .unsqueeze(0)

# create ones multiplier preds
ONES = torch.ones([1, IMAGE_SIZE[0], IMAGE_SIZE[1]])
ONES = torch.cat([ONES * -1, ONES * -1, ONES, ONES]).to(device=DEVICE,
                                                        dtype=torch.float)

HALVES = torch.ones([1, IMAGE_SIZE[0], IMAGE_SIZE[1]], dtype=torch.float,
                    device=DEVICE).repeat(4, 1, 1) * .5

# Then concatenate them on the 0th dimension
INDEX_TENSOR = torch.cat([x_index, y_index,
                          x_index, y_index]).to(device=DEVICE,
                                                dtype=torch.float)
