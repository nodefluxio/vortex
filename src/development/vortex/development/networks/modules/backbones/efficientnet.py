""" EfficientNet
Paper: EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946

Heavily inspired by Ross Wightman's definition
https://github.com/rwightman/pytorch-image-models/tree/master/timm/models

Hacked together by Vortex Team
"""

import re
import math
import torch.nn as nn
import numpy as np

from torch.hub import load_state_dict_from_url
from torch._six import container_abcs

from ..utils.layers import (
    DepthwiseSeparableConv, InvertedResidualBlock, 
    EdgeResidual
)
from ..utils.activations import get_act_layer
from ..utils.arch_utils import round_channels
from ..utils.layers import resolve_act_layer, resolve_norm_layer
from ..utils.conv2d import create_conv2d, CondConv2d
from .base_backbone import BackboneBase, BackboneConfig

from copy import deepcopy


_complete_url = lambda x: 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/' + x
default_cfgs = {
    'efficientnet_b0': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b0_ns-c0e6a31c.pth'), input_size=(3, 224, 224)
    ),
    'efficientnet_b1': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b1_ns-99dd0c41.pth'), input_size=(3, 240, 240)
    ),
    'efficientnet_b2': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b2_ns-00306e48.pth'), input_size=(3, 260, 260)
    ),
    'efficientnet_b3': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b3_ns-9d44bf68.pth'), input_size=(3, 300, 300)
    ),
    'efficientnet_b4': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b4_ns-d6313a46.pth'), input_size=(3, 380, 380)
    ),
    'efficientnet_b5': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b5_ns-6f26d0cf.pth'), input_size=(3, 456, 456)
    ),
    'efficientnet_b6': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b6_ns-51548356.pth'), input_size=(3, 528, 528)
    ),
    'efficientnet_b7': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_b7_ns-1dbc32de.pth'), input_size=(3, 600, 600)
    ),
    'efficientnet_b8': BackboneConfig(  ## this is actually lower than b5
        pretrained_url=_complete_url('tf_efficientnet_b8_ra-572d5dd9.pth'), input_size=(3, 672, 672)
    ),
    'efficientnet_l2': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_l2_ns-df73bb44.pth'), input_size=(3, 800, 800)
    ),
    'efficientnet_l2_475': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_l2_ns_475-bebbd00a.pth'), input_size=(3, 475, 475)
    ),
    'efficientnet_edge_s': BackboneConfig(pretrained_url=_complete_url('efficientnet_es_ra-f111e99c.pth')),
    'efficientnet_edge_m': BackboneConfig(
        pretrained_url=_complete_url('efficientnet_em_ra2-66250f76.pth'), input_size=(3, 240, 240)
    ),
    'efficientnet_edge_l': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_el-5143854e.pth'), input_size=(3, 300, 300),
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
    ),
    'efficientnet_lite0': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_lite0-0aa007d2.pth'),
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
    ),
    'efficientnet_lite1': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_lite1-bde8b488.pth'), input_size=(3, 240, 240),
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
    ),
    'efficientnet_lite2': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_lite2-dcccb7df.pth'), input_size=(3, 260, 260),
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)
    ),
    'efficientnet_lite3': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_lite3-b733e338.pth'), input_size=(3, 300, 300),
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)
    ),
    'efficientnet_lite4': BackboneConfig(
        pretrained_url=_complete_url('tf_efficientnet_lite4-741542c3.pth'), input_size=(3, 380, 380),
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)
    ),
}
""" Pretrained model URL
provided by `"rwightman/pytorch-image-models" <https://github.com/rwightman/pytorch-image-models>`_
only take the highest performing weight (if there are multiple weights)
for EfficientNet B0-B7 we use weight trained with NoisyStudent
"""

supported_models = list(default_cfgs.keys())

TF_BN_MOMENTUM = 1 - 0.99
TF_BN_EPSILON = 1e-3


class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, 
                 pad_type='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **_):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_channel, out_channel, kernel_size, stride=stride, 
            dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_channel, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class EfficientNetBuilder(nn.Module):
    def __init__(self, block_def, arch_params, global_params, stem_size=32,
                 fix_first_last_block=False, **kwargs):
        self.arch_params = arch_params
        self.block_def = block_def
        self.global_params = global_params
        self.fix_first_last_block = fix_first_last_block
        self.last_channel = stem_size
        self.out_channels = [stem_size]
        self.se_kwargs = global_params.pop('se_kwargs', None)

        self.blocks_args = self._decode_block_def()
        self.total_layers = sum(len(x) for x in self.blocks_args)
        self.layer_idx = 0

    def _round_channel(self, channel):
        channel_multiplier = self.arch_params[0]
        return round_channels(channel, channel_multiplier, 
            self.global_params['channel_divisor'], self.global_params['channel_min'])

    def _decode_block_def(self):
        block_def = self.block_def
        blocks_args = []
        for idx, stage_strings in enumerate(block_def):
            assert isinstance(stage_strings, container_abcs.Sequence)
            stage_args = [self._decode_str_def(stage_str) for stage_str in stage_strings]
            repeats = [arg.pop('repeat') for arg in stage_args]
            if not (self.fix_first_last_block and (idx == 0 or idx == len(block_def)-1)):
                stage_args = self._scale_stage_depth(stage_args, repeats)
            blocks_args.append(stage_args)
        return blocks_args

    def _decode_str_def(self, stage_str):
        assert isinstance(stage_str, str)
        stage_data = stage_str.split('_')
        args_map = {
            'r': ('repeat', int),
            'k': ('kernel_size', int),
            's': ('stride', int),
            'e': ('exp_ratio', float),
            'm': ('mid_channel', int),
            'c': ('out_channel', int),
            'se': ('se_ratio', float)
        }
        act_map = {
            're': 'relu',
            'r6': 'relu6',
            'hw': 'hard_swish',
            'sw': 'swish'
        }

        args = {'block_type': stage_data[0]}
        noskip = False
        for op in stage_data[1:]:
            if op == 'noskip':
                noskip = True
            elif op.startswith('n'):
                value = op[1:]
                if value in act_map:
                    value = get_act_layer(act_map[value])
                else:
                    value = get_act_layer(value)
                args['act_layer'] = value
            else:
                s = re.split(r'(\d.*)', op)
                assert len(s) >= 2
                key, cast = args_map[s[0]]
                args[key] = cast(s[1])
        args['noskip'] = noskip
        assert 'repeat' in args, "stage arguments does not have repeat ('r') argument"
        return args

    def _scale_stage_depth(self, stage_args, repeats):
        depth_multiplier = self.arch_params[1]
        num_repeat = sum(repeats)
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))
        repeats_scaled = []
        for r in reversed(repeats):
            rs = max(1, round(r/num_repeat * num_repeat_scaled))
            repeats_scaled.append(rs)
            num_repeat -= r
            num_repeat_scaled -= rs
        repeats_scaled = list(reversed(repeats_scaled))

        stage_args_scaled = []
        for sa, rep in zip(stage_args, repeats_scaled):
            stage_args_scaled.extend([deepcopy(sa) for _ in range(rep)])
        return stage_args_scaled

    def _make_layer(self, layer_args):
        assert isinstance(layer_args, dict)
        block_map = {
            'ds': DepthwiseSeparableConv,
            'ir': InvertedResidualBlock,
            'er': EdgeResidual,
            'cn': ConvBnAct
        }
        block_type = layer_args.pop('block_type')

        if block_type != 'cn' and 'drop_path_rate' in self.global_params:
            drop_rate = self.global_params['drop_path_rate'] * self.layer_idx / self.total_layers
            layer_args['drop_path_rate'] = drop_rate

        layer_args['in_channel'] = self.last_channel
        layer_args['out_channel'] = self._round_channel(layer_args['out_channel'])
        layer_args['pad_type'] = self.global_params['pad_type']
        layer_args['norm_layer'] = self.global_params['norm_layer']
        layer_args['norm_kwargs'] = self.global_params['norm_kwargs']
        layer_args['act_layer'] = self.global_params['act_layer']
        if 'mid_channel' in layer_args:
            layer_args['mid_channel'] = self._round_channel(layer_args['mid_channel'])
        if self.se_kwargs and block_type != 'cn':
            layer_args['se_kwargs'] = self.se_kwargs

        layer = block_map[block_type](**layer_args)

        self.last_channel = layer_args['out_channel']
        self.layer_idx += 1
        return layer

    def __call__(self):
        blocks = []
        for stage_args in self.blocks_args:
            assert isinstance(stage_args, list)
            stage = []
            for idx, args in enumerate(stage_args):
                assert args['stride'] in (1, 2)
                if idx > 0:
                    args['stride'] = 1
                layer = self._make_layer(args)
                stage.append(layer)
            blocks.append(nn.Sequential(*stage))
            self.out_channels.append(self.last_channel)
        return nn.Sequential(*blocks)


class EfficientNet(BackboneBase):
    def __init__(self, block_def, arch_params, global_params, num_classes=1000, in_channel=3,
                 stem_size=32, fix_stem=False, num_features=None, fix_block_first_last=False,
                 norm_layer=None, norm_kwargs=None, default_config=None):
        super(EfficientNet, self).__init__(default_config)
        assert isinstance(global_params, dict)

        if norm_layer is None:
            if 'norm_layer' in global_params:
                norm_layer = global_params['norm_layer']
            else:
                global_params['norm_layer'] = nn.BatchNorm2d
                norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            if 'norm_kwargs' in global_params and isinstance(norm_layer, nn.BatchNorm2d):
                norm_kwargs = global_params['norm_kwargs']
            else:
                global_params['norm_kwargs'] = {}
                norm_kwargs = {}

        self.in_channel = in_channel
        self.arch_params = arch_params
        self.block_def = block_def
        self.global_params = global_params
        self.num_features = self._round_channel(1280) if num_features is None else num_features

        act_layer = global_params['act_layer']
        pad_type = global_params['pad_type']

        if not fix_stem:
            stem_size = self._round_channel(stem_size)
        self.conv_stem = create_conv2d(in_channel, stem_size, 3, stride=2, 
            padding=pad_type, bias=False)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        blocks_builder = EfficientNetBuilder(block_def, arch_params, global_params, 
            stem_size=stem_size, fix_first_last_block=fix_block_first_last)
        self.blocks = blocks_builder()
        last_channel = blocks_builder.last_channel
        self.out_channels = blocks_builder.out_channels

        self.conv_head = create_conv2d(last_channel, self.num_features, 1, padding=pad_type, bias=False)
        self.bn2 = norm_layer(self.num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        dropout_rate = self.arch_params[3]
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(self.num_features, num_classes)
        self._num_classes = num_classes

        self._stages_channel = tuple(np.array(self.out_channels[1:])[[0,1,2,4,-1]])

        self.out_channels.extend([self.num_features, num_classes])
        effnet_init_weights(self)

    def _round_channel(self, channel):
        channel_multiplier = self.arch_params[0]
        channel_divisor = self.global_params['channel_divisor']
        channel_min = self.global_params['channel_min']
        return round_channels(channel, channel_multiplier, channel_divisor, channel_min)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def get_stages(self):
        """ get backbone stages

        This stage division is based on EfficientDet Implementation 
        (https://github.com/google/automl/tree/master/efficientdet),
        which takes the layers with spatial reduction of 2
        """
        if len(self.blocks) == 6:
            last_stage = self.blocks[5]
        elif len(self.blocks) == 7:
            last_stage = nn.Sequential(self.blocks[5], self.blocks[6])
        else:
            raise RuntimeError("Unable to get stages from efficientnet network, number of "
                "blocks in efficientnet should be 6 or 7, got {}".format(len(self.blocks)))
        stages = [
            nn.Sequential(
                self.conv_stem, self.bn1,
                self.act1, self.blocks[0]
            ),
            self.blocks[1],
            self.blocks[2],
            nn.Sequential(self.blocks[3], self.blocks[4]),
            last_stage
        ]
        return nn.Sequential(*stages)

    @property
    def stages_channel(self):
        return self._stages_channel

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_classifer_feature(self):
        return self.num_features

    def get_classifier(self):
        classifier = [
            self.conv_head, self.bn2, self.act2, self.global_pool,
            self.flatten, self.dropout, self.classifier
        ]
        return nn.Sequential(*classifier)

    def reset_classifier(self, num_classes, classifier = None):
        self._num_classes = num_classes
        if num_classes < 0:
            classifier = nn.Identity()
        elif classifier is None:
            classifier = nn.Linear(self.num_features, num_classes)
        if not isinstance(classifier, nn.Module):
            raise TypeError("'classifier' argument is required to have type of 'int' or 'nn.Module', "
                "got {}".format(type(classifier)))
        self.classifier = classifier


def effnet_init_weights(model, fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        model (nn.Module): model to initialize
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    for n, m in model.named_modules():
        if isinstance(m, CondConv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if fix_group_fanout:
                fan_out //= m.groups
            CondConv2d._initializer(m.weight, lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), 
                m.num_experts, m.weight_shape)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if fix_group_fanout:
                fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            fan_out = m.weight.size(0)  # fan-out
            fan_in = 0
            if 'routing_fn' in n:
                fan_in = m.weight.size(1)
            init_range = 1.0 / math.sqrt(fan_in + fan_out)
            m.weight.data.uniform_(-init_range, init_range)
            m.bias.data.zero_()


def _create_model(variant, block_def, global_params, arch_params, num_classes,
        override_params, pretrained, progress, **kwargs):
    assert isinstance(arch_params, container_abcs.Sequence), \
        "'arch_params' should be a sequence (e.g. list or tuple)"

    global_params.update({'drop_path_rate': 0.0})
    if override_params is not None:
        assert isinstance(override_params, container_abcs.Mapping), \
            "'override_params' should be a mapping (e.g. dict)"
        global_params.update(dict(override_params))

    if not pretrained:
        kwargs['num_classes'] = num_classes

    model = EfficientNet(block_def, arch_params, global_params, default_config=default_cfgs[variant], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(default_cfgs[variant].pretrained_url, progress=progress)
        model.load_state_dict(state_dict, strict=True)
        if num_classes != 1000:
            model.reset_classifier(num_classes)
    return model


def _efficientnet(variant, arch_params, num_classes=1000, override_params=None, 
                  pretrained=False, progress=True, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params (arch_params)
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    """
    block_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    global_params = {
        'channel_divisor': 8,
        'channel_min': None,
        'drop_path_rate': 0.2,
        'act_layer': resolve_act_layer(kwargs, default='swish'),
        'pad_type': 'same',
        'norm_layer': resolve_norm_layer(kwargs, default=nn.BatchNorm2d),
        'norm_kwargs': dict(eps=TF_BN_EPSILON, momentum=TF_BN_MOMENTUM)
    }

    model = _create_model(variant, block_def, global_params, arch_params, num_classes,
        override_params, pretrained, progress, **kwargs)
    return model


def _efficientnet_edge(variant, arch_params, num_classes=1000, override_params=None, 
                  pretrained=False, progress=True, **kwargs):
    """Creates an EfficientNet-EdgeTPU model

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu
    Blog post: https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    arch_params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-edge-s': (1.0, 1.0, 224, 0.2),    # edgetpu-S
    'efficientnet-edge-m': (1.0, 1.1, 240, 0.2),    # edgetpu-m
    'efficientnet-edge-l': (1.2, 1.4, 300, 0.3),    # edgetpu-l
    """
    block_def = [
        ['er_r1_k3_s1_e4_c24_m24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    global_params = {
        'channel_divisor': 8,
        'channel_min': None,
        'drop_path_rate': 0.2,
        'act_layer': resolve_act_layer(kwargs, default='relu'),
        'pad_type': 'same',
        'norm_layer': resolve_norm_layer(kwargs, default=nn.BatchNorm2d),
        'norm_kwargs': dict(eps=TF_BN_EPSILON, momentum=TF_BN_MOMENTUM)
    }

    model = _create_model(variant, block_def, global_params, arch_params, num_classes,
        override_params, pretrained, progress, **kwargs)
    return model


def _efficientnet_lite(variant, arch_params, num_classes=1000, override_params=None, 
                  pretrained=False, progress=True, **kwargs):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Blog post: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md

    arch_params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),
    """
    block_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    global_params = {
        'channel_divisor': 8,
        'channel_min': None,
        'drop_path_rate': 0.2,
        'act_layer': resolve_act_layer(kwargs, 'relu6'),
        'pad_type': 'same',
        'norm_layer': resolve_norm_layer(kwargs, default=nn.BatchNorm2d),
        'norm_kwargs': dict(eps=TF_BN_EPSILON, momentum=TF_BN_MOMENTUM)
    }
    kwargs['fix_stem'] = True
    kwargs['num_features'] = 1280
    kwargs['fix_block_first_last'] = True

    model = _create_model(variant, block_def, global_params, arch_params, num_classes,
        override_params, pretrained, progress, **kwargs)
    return model


def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B0 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b0', (1.0, 1.0, 224, 0.2),
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B1 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b1', (1.0, 1.1, 240, 0.2), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B2 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b2', (1.1, 1.2, 260, 0.3), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B3 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b3', (1.2, 1.4, 300, 0.3), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B4 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b4', (1.4, 1.8, 380, 0.4), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B5 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b5', (1.6, 2.2, 456, 0.4), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B6 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b6', (1.8, 2.6, 528, 0.5), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B7 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b7', (2.0, 3.1, 600, 0.5), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_b8(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-B8 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_b8', (2.2, 3.6, 672, 0.5), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_l2(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-L2 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_l2', (4.3, 5.3, 800, 0.5), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_l2_475(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-L2 with input size of 475 model from
    `"EfficientNet: Rethinking Model Scaling for CNNs" <https://arxiv.org/abs/1905.11946>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet('efficientnet_l2_475', (4.3, 5.3, 475, 0.5), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_edge_s(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-EdgeTPU-S model from
    `"EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML" 
    <https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    override_params = {'pad_type': ''}
    model = _efficientnet_edge('efficientnet_edge_s', (1.0, 1.0, 224, 0.2), override_params=override_params,
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_edge_m(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-EdgeTPU-M model from
    `"EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML" 
    <https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    override_params = {'pad_type': ''}
    model = _efficientnet_edge('efficientnet_edge_m', (1.0, 1.1, 240, 0.2), override_params=override_params,
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_edge_l(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-EdgeTPU-L model from
    `"EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML" 
    <https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet_edge('efficientnet_edge_l', (1.2, 1.4, 300, 0.3), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model


def efficientnet_lite0(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-Lite0 model from
    `"Original EfficientNet-Lite Implementation in Tensorflow" 
    <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet_lite('efficientnet_lite0', (1.0, 1.0, 224, 0.2), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_lite1(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-Lite1 model from
    `"Original EfficientNet-Lite Implementation in Tensorflow" 
    <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet_lite('efficientnet_lite1', (1.0, 1.1, 240, 0.2), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_lite2(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-Lite2 model from
    `"Original EfficientNet-Lite Implementation in Tensorflow" 
    <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet_lite('efficientnet_lite2', (1.1, 1.2, 260, 0.3), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_lite3(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-Lite3 model from
    `"Original EfficientNet-Lite Implementation in Tensorflow" 
    <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet_lite('efficientnet_lite3', (1.2, 1.4, 280, 0.3), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model

def efficientnet_lite4(pretrained=False, progress=True, **kwargs):
    r"""EfficientNet-Lite4 model from
    `"Original EfficientNet-Lite Implementation in Tensorflow" 
    <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _efficientnet_lite('efficientnet_lite4', (1.4, 1.8, 300, 0.3), 
        pretrained=pretrained, progress=progress, **kwargs)
    return model
