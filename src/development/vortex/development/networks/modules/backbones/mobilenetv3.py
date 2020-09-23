""" MobileNet V3
Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by / Copyright 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models/tree/master/timm/models

Edited by Vortex Team
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url
from torch._six import container_abcs

from .efficientnet import EfficientNetBuilder, effnet_init_weights
from ..utils.layers import resolve_act_layer, resolve_norm_args, resolve_norm_layer
from ..utils.arch_utils import round_channels
from ..utils.conv2d import create_conv2d
from ..utils.activations import get_act_fn
from .base_backbone import Backbone, ClassifierFeature

_complete_url = lambda x: 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/' + x
model_urls = {
    'mobilenetv3_large_075': _complete_url('tf_mobilenetv3_large_075-150ee8b0.pth'),
    'mobilenetv3_large_100': _complete_url('mobilenetv3_large_100_ra-f55367f5.pth'),
    'mobilenetv3_large_minimal_100': _complete_url('tf_mobilenetv3_large_minimal_100-8596ae28.pth'),
    'mobilenetv3_small_075': _complete_url('tf_mobilenetv3_small_075-da427f52.pth'),
    'mobilenetv3_small_100': _complete_url('tf_mobilenetv3_small_100-37f49e2b.pth'),
    'mobilenetv3_small_minimal_100': _complete_url('tf_mobilenetv3_small_minimal_100-922a7843.pth'),
    'mobilenetv3_rw': _complete_url('mobilenetv3_100-35495452.pth'),
}
supported_models = list(model_urls.keys())

TF_BN_MOMENTUM = 1 - 0.99
TF_BN_EPSILON = 1e-3


class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, block_def, arch_params, global_params, num_classes=1000, in_channel=3,
                 stem_size=16, num_features=1280, head_bias=True, dropout_rate=0., 
                 norm_layer=None, **kwargs):
        super(MobileNetV3, self).__init__()

        if norm_layer is None:
            if 'norm_layer' in global_params:
                norm_layer = global_params['norm_layer']
            else:
                global_params['norm_layer'] = nn.BatchNorm2d
                norm_layer = nn.BatchNorm2d

        self.num_classes = num_classes
        self.num_features = num_features
        self.block_def = block_def
        self.arch_params = arch_params

        norm_layer = global_params['norm_layer']
        norm_kwargs = global_params['norm_kwargs']
        act_layer = global_params['act_layer']
        pad_type = global_params['pad_type']

        # Stem
        stem_size = round_channels(stem_size, self.arch_params[0])
        self.conv_stem = create_conv2d(in_channel, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        global_params['channel_divisor'] = 8
        blocks_builder = EfficientNetBuilder(block_def, arch_params, global_params, stem_size=stem_size)
        self.blocks = blocks_builder()
        last_channel = blocks_builder.last_channel
        self.out_channels = blocks_builder.out_channels

        # Head + Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = create_conv2d(last_channel, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.out_channels.extend([self.num_features, num_classes])
        effnet_init_weights(self)

    def get_stages(self):
        out_channels = np.array(self.out_channels[:-2])
        layers = None
        if len(self.block_def) == 6:
            out_channels = out_channels[[0,1,2,4,6]]
            layers = [
                nn.Sequential(self.conv_stem, self.bn1, self.act1),
                self.blocks[0],
                self.blocks[1],
                nn.Sequential(self.blocks[2], self.blocks[3]),
                nn.Sequential(self.blocks[4], self.blocks[5])
            ]
        elif len(self.block_def) == 7:
            out_channels = out_channels[[1,2,3,5,7]]
            layers = [
                nn.Sequential(self.conv_stem, self.bn1, self.act1, self.blocks[0]),
                self.blocks[1],
                self.blocks[2],
                nn.Sequential(self.blocks[3], self.blocks[4]),
                nn.Sequential(self.blocks[5], self.blocks[6]),
            ]
        else:
            raise RuntimeError("Unknown MobileNetV3 block definition, the number of block "
                "definition should be 6 or 7, got {}".format(len(self.block_def)))
        return nn.Sequential(*layers), out_channels

    def get_classifier(self):
        classifier = [
            self.global_pool, self.conv_head, self.act2,
            self.flatten, self.dropout, self.classifier
        ]
        return nn.Sequential(*classifier)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.classifier(x)


def _create_model(variant, block_def, global_params, num_classes, channel_multiplier,
        override_params, pretrained, progress, **kwargs):

    if override_params is not None:
        assert isinstance(override_params, container_abcs.Mapping), \
            "'override_params' should be a mapping (e.g. dict)"
        global_params.update(dict(override_params))

    if not pretrained:
        kwargs['num_classes'] = num_classes

    drop_rate = 0.
    if 'dropout_rate' in kwargs:
        drop_rate = kwargs['dropout_rate']
    if 'drop_path_rate' in kwargs:
        global_params['drop_path_rate'] = kwargs.pop('drop_path_rate')
    arch_params = (channel_multiplier, 1.0, drop_rate, 224)
    model = MobileNetV3(block_def, arch_params, global_params, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[variant], progress=progress)
        model.load_state_dict(state_dict, strict=True)
        if num_classes != 1000:
            model.reset_classifier(num_classes)
    return model


def _mobilenet_v3_rw(variant, num_classes=1000, channel_multiplier=1.0, override_params=None, 
                  pretrained=False, progress=True, **kwargs):
    """Creates a MobileNet-V3 model.

    Paper: https://arxiv.org/abs/1905.02244
    """
    block_def = [
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]

    global_params = {
        'channel_divisor': 8,
        'channel_min': None,
        'act_layer': resolve_act_layer(kwargs, default='hard_swish'),
        'pad_type': kwargs.pop('pad_type', ''),
        'norm_layer': resolve_norm_layer(kwargs, default=nn.BatchNorm2d),
        'norm_kwargs': resolve_norm_args(kwargs),
        'se_kwargs': dict(gate_fn=get_act_fn('hard_sigmoid'), reduce_mid=True, divisor=1)
    }
    model_kwargs = dict(
        head_bias=False,
        **kwargs
    )

    model = _create_model(variant, block_def, global_params, num_classes, channel_multiplier,
        override_params, pretrained, progress, **model_kwargs)
    return model


def _mobilenet_v3(variant, num_classes=1000, channel_multiplier=1.0, override_params=None, 
                  pretrained=False, progress=True, **kwargs):
    """Creates a MobileNet-V3 model.

    Paper: https://arxiv.org/abs/1905.02244
    """
    act_layer_def = None
    block_def = None
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer_def = 'relu'
            block_def = [
                ['ds_r1_k3_s2_e1_c16'],
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                ['ir_r2_k3_s1_e3_c48'],
                ['ir_r3_k3_s2_e6_c96'],
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer_def = 'hard_swish'
            block_def = [
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer_def = 'relu'
            block_def = [
                ['ds_r1_k3_s1_e1_c16'],
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                ['ir_r3_k3_s2_e3_c40'],
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                ['ir_r2_k3_s1_e6_c112'],
                ['ir_r3_k3_s2_e6_c160'],
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer_def = 'hard_swish'
            block_def = [
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]

    global_params = {
        'channel_divisor': 8,
        'channel_min': None,
        'act_layer': resolve_act_layer(kwargs, default=act_layer_def),
        'pad_type': kwargs.pop('pad_type', ''),
        'norm_layer': resolve_norm_layer(kwargs, default=nn.BatchNorm2d),
        'norm_kwargs': resolve_norm_args(kwargs),
        'se_kwargs': dict(act_layer=nn.ReLU, gate_fn=get_act_fn('hard_sigmoid'), reduce_mid=True, divisor=8)
    }
    model_kwargs = dict(
        num_features=num_features,
        stem_size=16,
        **kwargs
    )

    model = _create_model(variant, block_def, global_params, num_classes, channel_multiplier,
        override_params, pretrained, progress, **model_kwargs)
    return model


def mobilenetv3_large_075(pretrained=False, progress=True, **kwargs):
    """MobileNetV3 Large 0.75 model from
    'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        kwargs['bn_eps'] = TF_BN_EPSILON
        kwargs['pad_type'] = 'same'
    model = _mobilenet_v3('mobilenetv3_large_075', channel_multiplier=0.75, 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def mobilenetv3_large_100(pretrained=False, progress=True, **kwargs):
    """MobileNetV3 Large 1.0 model from
    'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _mobilenet_v3('mobilenetv3_large_100', channel_multiplier=1.00, 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def mobilenetv3_small_075(pretrained=False, progress=True, **kwargs):
    """MobileNetV3 Small 0.75 model from
    'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        kwargs['bn_eps'] = TF_BN_EPSILON
        kwargs['pad_type'] = 'same'
    model = _mobilenet_v3('mobilenetv3_small_075', channel_multiplier=0.75, 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def mobilenetv3_small_100(pretrained=False, progress=True, **kwargs):
    """MobileNetV3 Small 1.0 model from
    'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        kwargs['bn_eps'] = TF_BN_EPSILON
        kwargs['pad_type'] = 'same'
    model = _mobilenet_v3('mobilenetv3_small_100', channel_multiplier=1.0, 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def mobilenetv3_rw(pretrained=False, progress=True, **kwargs):
    """MobileNetV3 RW from
    'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        kwargs['bn_eps'] = TF_BN_EPSILON
    model = _mobilenet_v3_rw('mobilenetv3_rw', channel_multiplier=1.0, 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def mobilenetv3_large_minimal_100(pretrained=False, progress=True, **kwargs):
    """MobileNetV3 Large 1.0 minimal version model from
    'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        kwargs['bn_eps'] = TF_BN_EPSILON
        kwargs['pad_type'] = 'same'
    model = _mobilenet_v3('mobilenetv3_large_minimal_100', channel_multiplier=1.0, 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def mobilenetv3_small_minimal_100(pretrained=False, progress=True, **kwargs):
    """MobileNetV3 Small 1.0 minimal version model from
    'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        kwargs['bn_eps'] = TF_BN_EPSILON
        kwargs['pad_type'] = 'same'
    model = _mobilenet_v3('mobilenetv3_small_minimal_100', channel_multiplier=1.0, 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def get_backbone(model_name: str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, **kwargs):
    if not model_name in supported_models:
        raise RuntimeError("model %s is not supported yet, available : %s" %(model_name, supported_models))

    kwargs['override_params'] = {
        'drop_path_rate': 0.0
    }
    network = eval('{}(pretrained=pretrained, num_classes=n_classes, **kwargs)'.format(model_name))
    stages, channels = network.get_stages()

    if feature_type == "tri_stage_fpn":
        backbone = Backbone(stages, channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(stages, network.get_classifier(), n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got %s" % (feature_type))
    return backbone
