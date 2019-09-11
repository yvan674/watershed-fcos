"""Watershed FCOS.

Hybrid using watershed energy map instead of centerness from FCOS.
"""
import torch.nn as nn
from .resnet import ResNet
from .fpn import FPN
from .wfcos_head import WFCOSHead

class WFCOS(nn.Module):
    def __init__(self, backbone_cfg, neck_cfg, head_cfg):
        """Initializes the watershed FCOS model

        Args:
            backbone_cfg (dict): The configuration of the backbone.
            neck_cfg (dict): The configuration of the neck.
            head_cfg (dict): The configuration of the head.
        """
        self.backbone = ResNet(depth=backbone_cfg['depth'],
                               num_stages=backbone_cfg['num_stages'],
                               out_indices=backbone_cfg['out_indices'],
                               frozen_stages=backbone_cfg['frozen_stages'],
                               norm_cfg=backbone_cfg['norm_cfg'],
                               style=backbone_cfg['style'])

        self.neck = FPN(
            in_channels=neck_cfg['in_channels'],
            out_channels=neck_cfg['out_channels'],
            start_level=neck_cfg['start_level'],
            add_extra_convs=neck_cfg['add_extra_convs'],
            extra_convs_on_inputs=neck_cfg['extra_convs_on_input'],
            num_outs=neck_cfg['num_outs'],
            relu_before_extra_convs=neck_cfg['relu_before_extra_convs']
        )

        self.head = WFCOSHead(
            num_classes=head_cfg['num_classes'],
            in_channels=head_cfg['in_channels'],
            max_energy=head_cfg['max_energy'],
            stacked_convs=head_cfg['stacked_convs'],
            feat_channels=head_cfg['feat_channels'],
            strides=head_cfg['strides']
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x
