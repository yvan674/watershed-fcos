"""Visualize.

Reads configuration from a JSON file and visualizes inference.
"""
import argparse

import json
import re
from os import makedirs
from os.path import exists, isdir

from utils.visualizer import Visualizer


def parse_args():
    description = "reads configuration from a supplied JSON file and runs the " \
                  "network training loop using the configurations."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('json_path', metavar='P', type=str,
                        help='path to the JSON file that contains the '
                             'configuration.')

    return parser.parse_args()


def parse_json(file_path):
    """Parses the JSON configuration string into a dictionary.

    The JSON file must include the following key-value pairs:
    - model (str): "FCOS" or "WFCOS". Chooses which model to use.
    - backbone:
        - pretrained (str): Path to the pretrained backbone model or an address.
        - depth (int): Backbone depth.
        - num_stages (int): Backbone number of stages.
        - out_indices (tuple): Out indices of each stage. Length must be equal
                               to cardinality of bb_num_stages.
        - frozen_stages (int): Frozen stages in the backbone.
        - norm_cfg (dict): Must contain "type", and "requires_grad". Determines
                           the configuration of the normalization layer used.
        - style (str): "pytorch" style or "caffe" style.
    - neck:
        - in_channels (tuple): The number of prediction channels for each
                               feature pyramid.
        - out_channels (int): The number of output channels.
        - start_level (int): The starting level of the FPN.
        - add_extra_convs (bool): Whether or not to add extra convolutions.
        - extra_convs_on_inputs (bool): Add convolutions to prediction or not.
        - num_outs (int): Number of output tensors.
        - relu_before_extra_convs (bool): Whether or not to add a relu layer.
    - head:
        - num_classes (int): Number of classes to output.
        - in_channels (int): Number of channels in the prediction of each head.
        - max_energy (int): Number of quantized energy units in the energy map.
        - stacked_convs (int): Number of stacked convolutions to include.
        - feat_channels (int): Number of channels in the stacked convolutions.
        - strides (tuple): Strides applied to each head. Length must be equal to
                           the num_outs value in neck.
        - split_convs (bool): Whether or not to split the stacked convolutions
                              for the classification and energy tensors.
    - loss:
        - classifier:
            - use_sigmoid (bool): Use sigmoidal focal loss or not.
            - gamma (float): Gamma value.
            - alpha (float): Alpha value.
            - loss_weight (float): Weighting of the loss.
        - bbox:
            - loss_weight (float): Weighting of the bounding box loss.
        - energy:
            - use_sigmoid (bool): Use sigmoidal cross entropy or not.
            - loss_weight (float): Weighting of the energy map loss.
    - optimizer:
        - type (str): 'SGD' or 'Adam'
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay.
        if 'SGD:
        - momentum (float): SGD Momentum
        - paramwise_options
            - bias_lr_mult (float)
            - bias_decay_mult (float)
        if 'Adam':
        - eps (float): Epsilon value.
    - lr_config:
        - warmup_iterations (int): Number of iterations to warmup with.
        - warmup_ratio (float or str): Value to use for the warmup. If a ratio
                                       or fraction is desired, a string can be
                                       used instead.
        - step (int or list): Period or milestones.
    - data
        - data_root (str): Path to coco dataset root.
        - imgs_per_gpu (int): Number of images to train per gpu per iteration.
        - train:
            - ann_file (str): Path to annotation file relative to data_root.
            - img_prefix (str): Path to images relative to data_root.
        - val:
            - ann_file (str): Path to annotation file relative to data_root.
            - img_prefix (str): Path to images relative to data_root.
        - test:
            - ann_file (str): Path to annotation file relative to data_root.
            - img_prefix (str): Path to images relative to data_root.
    - work_dir (str): Where to put logs and tensorflow/tensorboard stuff
    - checkpoint (str): Where to put the checkpoint file. This framework saves
                        checkpoints every epoch. When resuming, it will choose
                        the most recent checkpoint available if this points to a
                        dir. If a specific checkpoint is chosen, it will choose
                        that checkpoint and save further checkpoints to the same
                        directory.
    - total_epochs (int): Number of epochs to run for.
    - num_gpus (int): Number of GPUs to use. It can't be a float.
    - resume (bool): Whether or not to resume training from the most recent
                     checkpoint.
    - logging_level (str): Logging level to use.

    Args:
        file_path (str): The path to the JSON file.
    """
    with open(file_path, mode='r') as file:
        jc = json.loads(file.read())

    return jc


def main():
    args = parse_args()
    json_config = parse_json(args.json_path)

    # Make work_dir if it doesn't exist
    if not exists(json_config['work_dir']):
        makedirs(json_config['work_dir'])
    if isdir(json_config['checkpoint']) \
            and not exists(json_config['checkpoint']):
        makedirs(json_config['checkpoint'])

    visualizer = Visualizer(json_config)
    visualizer.run()


if __name__ == '__main__':
    main()
