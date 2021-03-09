from torch import nn
import torch
from .base_backbone import BackboneConfig, BackboneBase
from ..utils.arch_utils import load_pretrained
from ..utils.layers import make_divisible


__all__ = ['MobileNetV2', 'mobilenet_v2']


default_cfgs = {
    'mobilenet_v2': BackboneConfig(pretrained_url='https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'),
}

supported_models = [
    'mobilenet_v2'
]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, 
                 groups=1, norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            norm_layer(out_planes, **norm_kwargs),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1: # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, 
                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, 
                       norm_layer=norm_layer, norm_kwargs=norm_kwargs),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup, **norm_kwargs),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(BackboneBase):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, 
                 round_nearest=8, in_channel=3, norm_layer=None, norm_kwargs=None, default_config=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__(default_config)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            norm_kwargs = {}
        input_channel = in_channel
        stem_size = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        stem_size = make_divisible(stem_size * width_mult, round_nearest)
        self.last_channel = make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(input_channel, stem_size, stride=2, 
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)]
        input_channel = stem_size
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, 
                    stride, expand_ratio=t, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, 
            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._stages_channel = (stem_size, 24, 32, 96, 320)
        self._num_classes = num_classes

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def get_stages(self):
        stages = nn.Sequential(
            self.features[0],
            self.features[1:3],
            self.features[3:5],
            self.features[5:12],
            self.features[12:18]
        )
        return stages

    @property
    def stages_channel(self):
        return self._stages_channel

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_classifer_feature(self):
        return self.last_channel

    def get_classifier(self):
        return nn.Sequential(
            self.features[18],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            self.classifier
        )

    def reset_classifier(self, num_classes, classifier = None):
        self._num_classes = num_classes
        if num_classes < 0:
            classifier = nn.Identity()
        elif classifier is None:
            classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )
        if not isinstance(classifier, nn.Module):
            raise TypeError("'classifier' argument is required to have type of 'int' or 'nn.Module', "
                "got {}".format(type(classifier)))
        self.classifier = classifier


def mobilenet_v2(pretrained=False, progress=True, num_classes=1000, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if not pretrained:
        kwargs['num_classes'] = num_classes

    model = MobileNetV2(default_config=default_cfgs["mobilenet_v2"], **kwargs)
    if pretrained:
        load_pretrained(model, default_cfgs["mobilenet_v2"].pretrained_url, num_classes=num_classes, 
            first_conv_name="features.0.0", progress=progress)
    return model
