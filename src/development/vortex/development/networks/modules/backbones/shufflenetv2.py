import torch
import torch.nn as nn

from .base_backbone import Backbone, ClassifierFeature
from ..utils.arch_utils import load_pretrained

supported_models = [
    'shufflenetv2_x0.5',
    'shufflenetv2_x1.0',
    'shufflenetv2_x1.5',
    'shufflenetv2_x2.0',
]

__all__ = [
    'ShuffleNetV2',
    'shufflenetv2_x0_5',
    'shufflenetv2_x1_0',
    'shufflenetv2_x1_5',
    'shufflenetv2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, norm_layer=None, norm_kwargs=None):
        super(InvertedResidual, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            norm_kwargs = {}

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3,
                                    stride=self.stride, padding=1),
                norm_layer(inp, **norm_kwargs),
                nn.Conv2d(inp, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                norm_layer(branch_features, **norm_kwargs),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(branch_features, **norm_kwargs),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=3, stride=self.stride, padding=1),
            norm_layer(branch_features, **norm_kwargs),
            nn.Conv2d(branch_features, branch_features,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(branch_features, **norm_kwargs),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2Classifier(nn.Module):
    def __init__(self, out_size, n_classes):
        super(ShuffleNetV2Classifier, self).__init__()
        self.classifier = nn.Linear(out_size, n_classes)

    def forward(self, x):
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, 
                 in_channel=3, norm_layer=None, norm_kwargs=None):
        super(ShuffleNetV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            norm_kwargs = {}

        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = in_channel
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            norm_layer(output_channels, **norm_kwargs),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2, 
                norm_layer=norm_layer, norm_kwargs=norm_kwargs)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 
                    stride=1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            norm_layer(output_channels, **norm_kwargs),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(output_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

    def get_stages(self):
        channels = self._stage_out_channels.copy()[:4]
        channels.insert(0, channels[0])
        stages = [
            self.conv1,
            self.maxpool,
            self.stage2,
            self.stage3,
            self.stage4
        ]
        return nn.Sequential(*stages), channels

    def get_classifier(self):
        return nn.Sequential(
            self.conv5,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            self.fc
        )

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        in_channel = self._stage_out_channels[-1]
        self.fc = nn.Linear(in_channel, num_classes)


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    num_classes = 1000
    if pretrained and kwargs.get("num_classes", False):
        num_classes = kwargs.pop("num_classes")

    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                'pretrained {} is not supported as of now'.format(arch))
        load_pretrained(model, model_url, num_classes=num_classes, 
            first_conv_name="conv1", classifier_name="fc", progress=progress)
    return model


def shufflenetv2_x0_5(pretrained=False, progress=True, *args, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                          [4, 8, 4], [24, 48, 96, 192, 1024], *args, **kwargs)
    return model


def shufflenetv2_x1_0(pretrained=False, progress=True, *args, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                          [4, 8, 4], [24, 116, 232, 464, 1024], *args, **kwargs)
    return model


def shufflenetv2_x1_5(pretrained=False, progress=True, *args, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                          [4, 8, 4], [24, 176, 352, 704, 1024], *args, **kwargs)
    return model


def shufflenetv2_x2_0(pretrained=False, progress=True, *args, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                          [4, 8, 4], [24, 244, 488, 976, 2048], *args, **kwargs)
    return model


def get_backbone(model_name: str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, *args, **kwargs):
    if not model_name in supported_models:
        raise RuntimeError("model %s is not supported yet, "\
            "available : %s" % (model_name, supported_models))

    model_name = model_name.replace('.', '_')
    network = eval('{}(pretrained=pretrained, num_classes=n_classes, *args, **kwargs)'.format(model_name))
    stages, n_channels = network.get_stages()

    if feature_type == "tri_stage_fpn":
        backbone = Backbone(stages, n_channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(stages, network.get_classifier(), n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got %s" % (feature_type))
    return backbone
