import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import create_classifier


__all__ = ['SENet']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }


default_cfgs = {
    'legacy_senet154': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/legacy_senet154-e9eb9fe6.pth'),
    'legacy_seresnet18': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet18-4bb0ce65.pth',
        interpolation='bicubic'),
    'legacy_seresnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet34-a4004e63.pth'),
    'legacy_seresnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet50-ce0d4300.pth'),
    'legacy_seresnet101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet101-7e38fcc6.pth'),
    'legacy_seresnet152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet152-d17c99b7.pth'),
    'legacy_seresnext26_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26_32x4d-65ebdb501.pth',
        interpolation='bicubic'),
    'legacy_seresnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/legacy_se_resnext50_32x4d-f3651bad.pth'),
    'legacy_seresnext101_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/legacy_se_resnext101_32x4d-37725eac.pth'),
}


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)


class SEModule(nn.Module):

    def __init__(self, channels, reduction, return_sigmoid=False):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.return_sigmoid = return_sigmoid

    def forward(self, x, return_sigmoid=False):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        if self.return_sigmoid or return_sigmoid:
            return module_input * x, x
        else:
            return module_input * x


class SEModuleFULL(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModuleFULL, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        output = module_input + module_input * x
        return self.relu(output)


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(
            planes * 2, planes * 4, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, return_sigmoid=False):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction, return_sigmoid=return_sigmoid)
        self.downsample = downsample
        self.stride = stride
        self.return_sigmoid = return_sigmoid

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        if self.return_sigmoid:
            se_out, sigmoid = self.se_module(out)
            out = se_out + shortcut
            return self.relu(out), sigmoid
        else:
            out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out

class ResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, return_sigmoid=False):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.return_sigmoid = return_sigmoid

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu(out)

        return out


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out


class SENet(nn.Module):

    def __init__(
            self, block, layers, groups, reduction, drop_rate=0.2,
            in_chans=3, inplanes=64, input_3x3=False, downsample_kernel_size=1,
            downsample_padding=0, return_sigmoid= False, num_classes=1000, global_pool='avg'):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.return_sigmoid = return_sigmoid
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(
                    in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        self.pool0 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='layer0')]
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.feature_info += [dict(num_chs=64 * block.expansion, reduction=4, module='layer1')]
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.feature_info += [dict(num_chs=128 * block.expansion, reduction=8, module='layer2')]
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.feature_info += [dict(num_chs=256 * block.expansion, reduction=16, module='layer3')]
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.feature_info += [dict(num_chs=512 * block.expansion, reduction=32, module='layer4')]
        self.num_features = 512 * block.expansion
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size,
                    stride=stride, padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, groups, reduction))
        layers.append(block(self.inplanes, planes, groups, reduction,
                            return_sigmoid=self.return_sigmoid))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^layer0', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.layer0(x)
        x = self.pool0(x)
        if self.return_sigmoid:
            x, sigmoid1 = self.layer1(x)
            x, sigmoid2 = self.layer2(x)
            x, sigmoid3 = self.layer3(x)
            x, sigmoid4 = self.layer4(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def get_sigmoid(self, x):
        x = self.layer0(x)
        x = self.pool0(x)
        x, sigmoid1 = self.layer1(x)
        x, sigmoid2 = self.layer2(x)
        x, sigmoid3 = self.layer3(x)
        x, sigmoid4 = self.layer4(x)
        return [sigmoid1, sigmoid2, sigmoid3, sigmoid4]

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

class SENetResNet(nn.Module):

    def __init__(
            self, block, layers, groups, reduction, drop_rate=0.2,
            in_chans=3, inplanes=64, input_3x3=False, downsample_kernel_size=1,
            downsample_padding=0, return_sigmoid= False, num_classes=1000, global_pool='avg'):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENetResNet, self).__init__()
        #del self.layer0
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.return_sigmoid = return_sigmoid
        self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        self.pool0 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='layer0')]
        self.layer1 = self._make_layer(
            ResNetBottleneck,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.se_module1 = SEModule(channels=64* block.expansion, reduction=4)
        #self.downsample = downsample
        self.feature_info += [dict(num_chs=64 * block.expansion, reduction=4, module='layer1')]
        self.layer2 = self._make_layer(
            ResNetBottleneck,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.se_module2 = SEModule(channels=128* block.expansion, reduction=8)
        self.feature_info += [dict(num_chs=128 * block.expansion, reduction=8, module='layer2')]
        self.layer3 = self._make_layer(
            ResNetBottleneck,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.se_module3 = SEModule(channels=256* block.expansion, reduction=16)
        self.feature_info += [dict(num_chs=256 * block.expansion, reduction=16, module='layer3')]
        self.layer4 = self._make_layer(
            ResNetBottleneck,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.se_module4 = SEModule(channels=512* block.expansion, reduction=32)
        self.feature_info += [dict(num_chs=512 * block.expansion, reduction=32, module='layer4')]
        self.num_features = 512 * block.expansion
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            _weight_init(m)
    def get_sigmoid(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x, sig1 = self.se_module1(x, return_sigmoid=True)
        x = self.layer2(x)
        x, sig2 = self.se_module2(x, return_sigmoid=True)
        x = self.layer3(x)
        x, sig3 = self.se_module3(x, return_sigmoid=True)
        x = self.layer4(x)
        _, sig4 = self.se_module4(x, return_sigmoid=True)
        return [sig1, sig2, sig3, sig4]

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool0(x)
        if self.return_sigmoid:
            x, sigmoid1 = self.layer1(x)
            x, sigmoid2 = self.layer2(x)
            x, sigmoid3 = self.layer3(x)
            x, sigmoid4 = self.layer4(x)
        else:
            x = self.layer1(x)
            x = self.se_module1(x)
            x = self.layer2(x)
            x = self.se_module2(x)
            x = self.layer3(x)
            x = self.se_module3(x)
            x = self.layer4(x)
            x = self.se_module4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.last_linear(x)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size,
                    stride=stride, padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, groups, reduction))
        layers.append(block(self.inplanes, planes, groups, reduction,
                            return_sigmoid=self.return_sigmoid))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^layer0', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)


def _create_senet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(SENet, variant, pretrained, **kwargs)

def legacy_seresnet50(return_sigmoid=False, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1,
        reduction=16, return_sigmoid=return_sigmoid, **kwargs)
    return SENet(**model_args)#_create_senet('legacy_seresnet50', pretrained, **model_args)

def legacy_seresnet50_custom(return_sigmoid=False, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1,
        reduction=16, return_sigmoid=return_sigmoid, **kwargs)
    return SENetResNet(**model_args)#_create_senet('legacy_seresnet50', pretrained, **model_args)
