import math
import torch
from typing import Optional
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import timm
from torch import nn
from torchvision import models
from torchvision.models import _utils
from detectron2.layers import Conv2d, ShapeSpec, get_norm, FrozenBatchNorm2d
from detectron2.modeling import build_resnet_backbone
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from src.layers import FrozenBatchNormAct2d
from .convnext import convnext_base, convnext_base_se
from .senet import SEModuleFULL
import os

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

class FPN_RES(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, backbone_type, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN_RES, self).__init__()
        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        strides = [4, 8, 16, 32]
        assert backbone_type in ["res", "res18", "eff_b2", "eff_b0",
                                 "convnext", "inception", "mobilenet_v2",
                                 "vgg16"]
        if backbone_type == "res":
            in_channels_per_feature = [256, 512, 1024, 2048]
            self.func_bottom_up = self.extract_bottom_res
        elif backbone_type == "res18":
            in_channels_per_feature = [64, 128, 256, 512]
            self.func_bottom_up = self.extract_bottom_res
        elif backbone_type == "eff_b2":
            in_channels_per_feature = [24, 48, 120, 352]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == "eff_b0":
            in_channels_per_feature = [24, 40, 112, 320]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == "inception":
            in_channels_per_feature = [192, 320, 1088, 1536]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == "mobilenet_v2":
            in_channels_per_feature = [24, 32, 96, 320]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == "mobilenet_v3":
            in_channels_per_feature = [24, 32, 96, 320]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == 'convnext':
            in_channels_per_feature = [128, 256, 512, 1024]
            self.func_bottom_up = self.extract_bottom_convnext

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []
        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def extract_bottom_res(self, x):
        return self.bottom_up(x)
    def extract_bottom_vgg(self, x):
        feats = self.bottom_up(x)[2:]
        return {k: feat for k, feat in zip(self.in_features, feats)}
    def extract_bottom_eff(self, x):
        feats = self.bottom_up(x)[1:]
        return {k: feat for k, feat in zip(self.in_features, feats)}
    def extract_bottom_convnext(self, x):
        feats = self.bottom_up(x)
        return {k: feat for k, feat in zip(self.in_features, feats)}

    def forward(self, x, stop_grad=False):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.func_bottom_up(x)
        if stop_grad:
            bottom_up_features = {key: value.detach() for key, value in bottom_up_features.items()}
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for features, lateral_conv, output_conv in zip(
            self.in_features[-2::-1], self.lateral_convs[1:], self.output_convs[1:]
        ):
            features = bottom_up_features[features]
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
            # Has to use explicit forward due to https://github.com/pytorch/pytorch/issues/47336
            lateral_features = lateral_conv.forward(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv.forward(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class SumCell(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, with_out_conv=True):
        super(SumCell, self).__init__()
        dict_convs = dict(groups=1, kernel_size=3, padding=1, bias=True, norm=norm_cfg)
        self.with_out_conv = with_out_conv
        ## relu => conv > norm
        if with_out_conv:
            self.out_conv = Conv2d(
                in_channels, out_channels, **dict_convs)
        self.upsample_mode = 'nearest'
        self.activation = F.relu

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)
        else:
            if x.shape[-2] % size[-2] != 0 or x.shape[-1] % size[-1] != 0:
                h, w = x.shape[-2:]
                target_h, target_w = size
                pad_h = math.ceil(h / target_h) * target_h - h
                pad_w = math.ceil(w / target_w) * target_w - w
                pad_l = pad_w // 2
                pad_r = pad_w - pad_l
                pad_t = pad_h // 2
                pad_b = pad_h - pad_t
                pad = (pad_l, pad_r, pad_t, pad_b)
                x = F.pad(x, pad, mode='constant', value=0.0)
            kernel_size = (x.shape[-2] // size[-2], x.shape[-1] // size[-1])
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            return x

    def _binary_op(self, x1, x2):
        return x1 + x2
    def forward(self, x1, x2, out_size):
        if out_size is None:  # resize to larger one
            out_size = max(x1.size()[2:], x2.size()[2:])
        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)
        x = self._binary_op(x1, x2)
        if self.with_out_conv:
            x = self.activation(x)
            x = self.out_conv(x)
        return x

class GlobalPoolingCell(SumCell):

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1
class NASFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, backbone_type, in_features, out_channels,
            norm="", top_block=None, fuse_type="sum", stack_times=7,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(NASFPN, self).__init__()
        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        strides = [4, 8, 16, 32]
        norm = 'BN'
        assert backbone_type in ["res", "res18", "eff_b2", "eff_b0",
                                 "convnext", "inception", "mobilenet_v2",
                                 "vgg16"]
        if backbone_type == "res":
            in_channels_per_feature = [256, 512, 1024, 2048]
            self.func_bottom_up = self.extract_bottom_res
        elif backbone_type == "res18":
            in_channels_per_feature = [64, 128, 256, 512]
            self.func_bottom_up = self.extract_bottom_res
        elif backbone_type == "eff_b2":
            in_channels_per_feature = [24, 48, 120, 352]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == "eff_b0":
            in_channels_per_feature = [24, 40, 112, 320]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == "mobilenet_v2":
            in_channels_per_feature = [24, 32, 96, 320]
            self.func_bottom_up = self.extract_bottom_eff
        elif backbone_type == 'convnext':
            in_channels_per_feature = [128, 256, 512, 1024]
            self.func_bottom_up = self.extract_bottom_convnext

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            weight_init.c2_xavier_fill(lateral_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            lateral_convs.append(lateral_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs#[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
        extra_conv = Conv2d(
            out_channels, out_channels, 1, norm=get_norm(norm, out_channels))
        self.extra_downsample = nn.Sequential(extra_conv, nn.MaxPool2d(2, 2))
        self.stack_times = stack_times
        # add NAS FPN connections
        self.fpn_stages = nn.ModuleList()

        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_cfg=get_norm(norm ,out_channels))
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_cfg=get_norm(norm ,out_channels))
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_cfg=get_norm(norm, out_channels))
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_cfg=get_norm(norm, out_channels))
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_55_5'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_cfg=get_norm(norm, out_channels))
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_77_7'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_cfg=get_norm(norm, out_channels))
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_cfg=get_norm(norm, out_channels))
            self.fpn_stages.append(stage)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def extract_bottom_res(self, x):
        return self.bottom_up(x)
    def extract_bottom_vgg(self, x):
        feats = self.bottom_up(x)[2:]
        return {k: feat for k, feat in zip(self.in_features, feats)}
    def extract_bottom_eff(self, x):
        feats = self.bottom_up(x)[1:]
        return {k: feat for k, feat in zip(self.in_features, feats)}
    def extract_bottom_convnext(self, x):
        feats = self.bottom_up(x)
        return {k: feat for k, feat in zip(self.in_features, feats)}

    def forward(self, x, stop_grad=False):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.func_bottom_up(x)
        if stop_grad:
            bottom_up_features = {key: value.detach() for key, value in bottom_up_features.items()}
        results = []

        feats = [
            lateral_conv(bottom_up_features[self.in_features[i]])
            for i, lateral_conv in enumerate(self.lateral_convs)]
        #for downsample in self.extra_downsamples:
        feats.append(self.extra_downsample(feats[-1]))
        p3, p4, p5, p6, p7 = feats
        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])
        #assert len(self._out_features) == len(results)
        results = [p3, p4, p5, p6, p7]
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class SENASFPN(NASFPN):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, *args, **kwargs):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(SENASFPN, self).__init__(*args, **kwargs)

        self.se_modules = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            se_module = SEModuleFULL(lateral_conv.out_channels, 16)
            self.se_modules.append(se_module)
            self.add_module("fpn_lateral_se{}".format(i), se_module)

    def forward(self, x, stop_grad=False):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.func_bottom_up(x)
        if stop_grad:
            bottom_up_features = {key: value.detach() for key, value in bottom_up_features.items()}
        results = []

        feats = [
            se_module(lateral_conv(bottom_up_features[self.in_features[i]]))
            for i, (lateral_conv, se_module) in enumerate(zip(self.lateral_convs, self.se_modules))]
        #for downsample in self.extra_downsamples:
        feats.append(self.extra_downsample(feats[-1]))
        p3, p4, p5, p6, p7 = feats
        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])
        #assert len(self._out_features) == len(results)
        results = [p3, p4, p5, p6, p7]
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }



@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone_caffe(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    back_class = eval(cfg.MODEL.FPN.NAME)
    backbone = back_class(
        bottom_up=bottom_up,
        backbone_type="res",
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone_efficient(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    map_model = {'efficientnet_b2': ["efficientnet_b2_ra-bcdf34b7.pth", "eff_b2"],
                 'tf_efficientnet_b0_ns': ["tf_efficientnet_b0_ns-c0e6a31c.pth", "eff_b0"],
                 'tf_efficientnet_b2_ns': ["tf_efficientnet_b2_ns-00306e48.pth", "eff_b2"]}
    model_name = cfg.MODEL.PRETRAIN_NAME
    model = timm.create_model(model_name, pretrained=False, features_only=True)
    dicts = torch.load(os.path.join(os.path.abspath(os.getcwd()), 'pre_models/', map_model[model_name][0]))
    model.load_state_dict(dicts, strict=False)
    frozen_range = [model.conv_stem, model.blocks[0]]
    model(torch.rand(2, 3, 224, 224))
    for module in frozen_range:
        for param in module.named_parameters():
            param[1].requires_grad = False
    model = FrozenBatchNormAct2d.convert_frozen_batchnorm(model)
    bottom_up = model
    bottom_up(torch.rand(2, 3, 224, 224))
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    back_class = eval(cfg.MODEL.FPN.NAME)
    backbone = back_class(
        bottom_up=bottom_up,
        backbone_type=map_model[model_name][1],
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone_mobilenet(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    model_name = cfg.MODEL.PRETRAIN_NAME
    dir_path = os.path.join(os.path.abspath(os.getcwd()), 'pre_models/')
    if 'mobilenet_v2' in model_name:
        dicts = torch.load(os.path.join(dir_path, "mobilenetv2_100_ra-b33bc2c4.pth"))
        model = timm.create_model("mobilenetv2_100", pretrained=False, features_only=True)
    model.load_state_dict(dicts, strict=False)
    frozen_range = [model.conv_stem]
    model(torch.rand(2, 3, 224, 224))
    for module in frozen_range:
        for param in module.named_parameters():
            param[1].requires_grad = False
    model = FrozenBatchNormAct2d.convert_frozen_batchnorm(model)
    bottom_up = model
    bottom_up(torch.rand(2, 3, 224, 224))
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    back_class = eval(cfg.MODEL.FPN.NAME)
    backbone = back_class(
        bottom_up=bottom_up,
        backbone_type=model_name,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone_resnet(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    return_layers = {"layer1": "res2", "layer2": "res3",
                     "layer3": "res4", "layer4": "res5"}
    resnet = models.resnet50(pretrained=False)
    dir_path = os.path.join(os.path.abspath(os.getcwd()), 'pre_models/')
    backbone_type = 'res'
    if "augmix" in cfg.MODEL.PRETRAIN_NAME:
        checkpoint = torch.load(os.path.join(dir_path, "augmix_resnet.pth.tar"))
        dicts = checkpoint['state_dict']
    elif 'resnet18' in cfg.MODEL.PRETRAIN_NAME:
        dicts = torch.load(os.path.join(dir_path, "semi_weakly_supervised_resnet18-118f1556.pth"))
        resnet = models.resnet18(pretrained=False)
        backbone_type = 'res18'
    elif 'ssl_res' in cfg.MODEL.PRETRAIN_NAME:
        dicts = torch.load(os.path.join(dir_path, "semi_supervised_resnet50-08389792.pth"))
    elif "fbnet" in cfg.MODEL.PRETRAIN_NAME:
        dicts = torch.load(os.path.join(dir_path, "semi_weakly_supervised_resnet50-16a12f1b.pth"))
    elif "resnext_swsl" in cfg.MODEL.PRETRAIN_NAME:
        dicts = torch.load(os.path.join(dir_path, "semi_weakly_supervised_resnext50_32x4-72679e44.pth"))
        resnet = timm.create_model("resnext50_32x4d", pretrained=False, features_only=False)
    elif "senet" in cfg.MODEL.PRETRAIN_NAME:
        dicts = torch.load(os.path.join(dir_path, "seresnet50_ra_224-8efdb4bb.pth"))
        resnet = timm.create_model("seresnet50", pretrained=False, features_only=False)

    new_dict = {}
    for key in dicts.keys():
        new_dict[key.replace("module.", "")] = dicts[key]
    resnet.load_state_dict(new_dict, strict=False)
    frozen_range = [resnet.conv1, resnet.layer1]
    resnet(torch.rand(2, 3, 224, 224))
    for module in frozen_range:
        for param in module.named_parameters():
            param[1].requires_grad = False
    resnet = FrozenBatchNorm2d.convert_frozen_batchnorm(resnet)
    bottom_up = _utils.IntermediateLayerGetter(resnet, return_layers)
    bottom_up(torch.rand(2, 3, 224, 224))
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    back_class = eval(cfg.MODEL.FPN.NAME)
    if back_class == NASFPN or back_class == SENASFPN:
        input_dicts = dict(bottom_up=bottom_up,
                           backbone_type=backbone_type,
                           in_features=in_features,
                           out_channels=out_channels,
                           norm=cfg.MODEL.FPN.NORM,
                           top_block=LastLevelMaxPool(),
                           fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
                           stack_times=cfg.MODEL.FPN.NAS_STACK)
    else:
        input_dicts = dict(bottom_up=bottom_up,
                           backbone_type=backbone_type,
                           in_features=in_features,
                           out_channels=out_channels,
                           norm=cfg.MODEL.FPN.NORM,
                           top_block=LastLevelMaxPool(),
                           fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
                           )
    backbone = back_class(**input_dicts)
    return backbone

@BACKBONE_REGISTRY.register()
def build_eval_senet_fpn_backbone(cfg, input_shape: ShapeSpec):
    dicts = torch.load(os.path.join(os.path.abspath(os.getcwd()), 'pre_models/',
                                    "seresnet50_ra_224-8efdb4bb.pth"))
    from .senet import legacy_seresnet50
    resnet = legacy_seresnet50()
    resnet(torch.rand(2, 3, 224, 224))
    new_dict = {}
    for key in dicts.keys():
        new_dict[key.replace("module.", "")] = dicts[key]
    resnet.load_state_dict(new_dict, strict=False)
    bottom_up = FrozenBatchNorm2d.convert_frozen_batchnorm(resnet)
    bottom_up(torch.rand(2, 3, 224, 224))
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    back_class = eval(cfg.MODEL.FPN.NAME)
    backbone = back_class(
        bottom_up=bottom_up,
        backbone_type = "res",
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_convnext_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, mu
        st be a subclass of :class:`Backbone`.
    """
    if "in1k" in cfg.MODEL.PRETRAIN_NAME:
        path_model = "convnext_base_1k_224_ema.pth"
        model = convnext_base(pretrained=False, in_22k=False)
    elif "senet" in cfg.MODEL.PRETRAIN_NAME:
        path_model = "convnext_base_22k_224.pth"
        model = convnext_base_se(pretrained=False, in_22k=True)
    else:
        path_model = "convnext_base_22k_224.pth"
        model = convnext_base(pretrained=False, in_22k=True)
    dicts = torch.load(os.path.join(os.path.abspath(os.getcwd()), 'pre_models/',
                                    path_model))
    model.load_state_dict(dicts["model"], strict=False)

    frozen_range = [model.downsample_layers[0], model.stages[0]]
    for module in frozen_range:
        for param in module.named_parameters():
            param[1].requires_grad = False
    for param in model.named_parameters():
        if 'norm' in param[0]:
            param[1].requires_grad = False
    model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    bottom_up = model
    bottom_up(torch.rand(2, 3, 224, 224))
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    back_class = eval(cfg.MODEL.FPN.NAME)
    backbone = back_class(
        bottom_up=bottom_up,
        backbone_type="convnext",
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_fpn_backbone_selike_resnet(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    return_layers = {"se_module1": "res2", "se_module2": "res3",
                     "se_module3": "res4", "se_module4": "res5"}
    from .senet import legacy_seresnet50_custom
    resnet = legacy_seresnet50_custom()
    if "fbnet" in cfg.MODEL.PRETRAIN_NAME:
        dicts = torch.load(os.path.join(os.path.abspath(os.getcwd()), 'pre_models/',
                                        "semi_weakly_supervised_resnet50-16a12f1b.pth"))
    new_dict = {}
    for key in dicts.keys():
        new_dict[key.replace("module.", "")] = dicts[key]
    resnet.load_state_dict(new_dict, strict=False)
    frozen_range = [resnet.conv1, resnet.layer1]
    resnet(torch.rand(2, 3, 224, 224))
    for module in frozen_range:
        for param in module.named_parameters():
            param[1].requires_grad = False
    resnet = FrozenBatchNorm2d.convert_frozen_batchnorm(resnet)
    bottom_up = _utils.IntermediateLayerGetter(resnet, return_layers)
    bottom_up(torch.rand(2, 3, 224, 224))
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    back_class = eval(cfg.MODEL.FPN.NAME)
    backbone = back_class(
        bottom_up=bottom_up,
        backbone_type = "res",
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone