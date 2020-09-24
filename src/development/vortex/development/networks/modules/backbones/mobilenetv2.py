from torch import nn
import torch
from .base_backbone import Backbone, ClassifierFeature
from ..utils.arch_utils import load_pretrained
from ..utils.layers import make_divisible


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
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


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, 
                 round_nearest=8, in_channel=3, norm_layer=None, norm_kwargs=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
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
        channels = [16, 24, 32, 96, 320]
        return stages, channels

    def get_classifier(self):
        return nn.Sequential(
            self.features[18],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            self.classifier
        )

    def reset_classifier(self, num_classes):
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )


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

    model = MobileNetV2(**kwargs)
    if pretrained:
        load_pretrained(model, model_urls["mobilenet_v2"], num_classes=num_classes, 
            first_conv_name="features.0.0", progress=progress)
    return model


def get_backbone(model_name: str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, *args, **kwargs):
    if not (model_name in supported_models):
        raise RuntimeError("unsupported model: %s; supported in mobilenetv2: %s" %
                           (model_name, supported_models))
    model = mobilenet_v2(pretrained=pretrained, num_classes=n_classes, *args, **kwargs)
    stages, channels = model.get_stages()

    if feature_type == "tri_stage_fpn":
        backbone = Backbone(stages, channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(stages, model.get_classifier(), n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got %s" % (feature_type))
    return backbone
