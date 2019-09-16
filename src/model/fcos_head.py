"""FCOS Head.

Head from FCOS architecture.
"""
import torch.nn as nn
from .utils.conv_module import ConvModule
from .utils.weight_init import normal_init, bias_init_with_prob
from .utils.scale import Scale
from .utils import multi_apply


INF=1e8


class FCOSHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 conv_cfg=None,
                 norm_cfg=None):
        """
        Creates a head based on FCOS that uses an energies map, not centerness
        Args:
            num_classes (int): Number of classes to output.
            in_channels (int): Number of innput channels.
            feat_channels (int): Number of feature channels in each of the
                stacked convolutions.
            stacked_convs (int): Number of stacked convolutions to have.
            strides (tuple): Stride value for each of the heads.
            regress_ranges (tuple): The regression range for each of the heads.
            loss_cls (dict): A description of the loss to use for the
                classfication output.
            loss_bbox (dict): A description of the loss to use for the bbox
                output.
            loss_energy (dict): A description of the loss to use for the energies
                map output.
            conv_cfg (dict): A description of the configuration of the
                convolutions in the stacked convolution.
            norm_cfg (dict): A description of the normalization configuration of
                the layers of the stacked convolution.
        """
        super(FCOSHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()


    def _init_layers(self):
        """Initialize each of the layers needed."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])


    def init_weights(self):
        """Initialize the weights for all the layers with a normal dist."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)


    def forward(self, feats):
        """Run forwards on the network.

        Args:
            feats (tuple): tuple of torch tensors handed off from the neck.
                Expects the use of FPN as the neck, giving multiple feature
                tensors.

        Returns:
            (tuple): A tuple of 3-tuples of tensors, each 3-tuple representing
                cls_score, bbox_pred, energies of the different feature layers.
        """
        # Use a multi_apply function to run forwards on each feats tensor
        return multi_apply(self.forward_single, feats, self.scales)


    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness
