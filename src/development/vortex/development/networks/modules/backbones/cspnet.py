"""PyTorch CspNet

A PyTorch implementation of Cross Stage Partial Networks including:
* CSPResNet50
* CSPResNeXt50
* CSPDarkNet53

Based on paper `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929

Reference impl via darknet cfg files at https://github.com/WongKinYiu/CrossStagePartialNetworks

Hacked together by / Copyright 2020 Ross Wightman
from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cspnet.py

Modified by Vortex Team
"""

import torch
import torch.nn as nn

from .base_backbone import Backbone, ClassifierFeature
from ..utils.layers import ClassifierHead, ConvBnAct, DropPath
from ..utils.arch_utils import load_pretrained


_complete_url = lambda x: 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/' + x
model_urls = {
    'cspresnet50': _complete_url('cspresnet50_ra-d3e8d487.pth'),
    'cspresnext50': _complete_url('cspresnext50_ra_224-648b4713.pth'),
    'cspdarknet53': _complete_url('cspdarknet53_ra_256-d05c7c21.pth'),
}
supported_models = list(model_urls.keys())

model_cfgs = dict(
    cspresnet50=dict(
        stem=dict(out_chs=64, kernel_size=7, stride=2, pool='max'),
        stage=dict(
            out_chs=(128, 256, 512, 1024),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            exp_ratio=(2.,) * 4,
            bottle_ratio=(0.5,) * 4,
            block_ratio=(1.,) * 4,
            cross_linear=True,
        )
    ),
    cspresnext50=dict(
        stem=dict(out_chs=64, kernel_size=7, stride=2, pool='max'),
        stage=dict(
            out_chs=(256, 512, 1024, 2048),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            groups=(32,) * 4,
            exp_ratio=(1.,) * 4,
            bottle_ratio=(1.,) * 4,
            block_ratio=(0.5,) * 4,
            cross_linear=True,
        )
    ),
    cspdarknet53=dict(
        stem=dict(out_chs=32, kernel_size=3, stride=1, pool=''),
        stage=dict(
            out_chs=(64, 128, 256, 512, 1024),
            depth=(1, 2, 8, 8, 4),
            stride=(2,) * 5,
            exp_ratio=(2.,) + (1.,) * 4,
            bottle_ratio=(0.5,) + (1.0,) * 4,
            block_ratio=(1.,) + (0.5,) * 4,
            down_growth=True,
        )
    ),
)


def create_stem(in_chans=3, out_chs=32, kernel_size=3, stride=2, pool='',
        act_layer=None, norm_layer=None, norm_kwargs=None, aa_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    norm_kwargs = norm_kwargs or {}

    stem = nn.Sequential()
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    assert len(out_chs)
    in_c = in_chans
    last_conv = ''
    for i, out_c in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        stem.add_module(conv_name, ConvBnAct(
            in_c, out_c, kernel_size, stride=stride if i == 0 else 1,
            act_layer=act_layer, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        )
        in_c = out_c
        last_conv = conv_name
    if pool:
        if aa_layer is not None:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            stem.add_module('aa', aa_layer(channels=in_c, stride=2))
        else:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return stem, dict(num_chs=in_c, reduction=stride, module='.'.join(['stem', last_conv]))


class ResBottleneck(nn.Module):
    """ ResNe(X)t Bottleneck Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.25, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 aa_layer=None, drop_block=None, drop_path=None):
        super(ResBottleneck, self).__init__()
        norm_kwargs = norm_kwargs or None
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, norm_kwargs=norm_kwargs, 
            aa_layer=aa_layer, drop_block=drop_block)

        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups, **ckwargs)
        self.conv3 = ConvBnAct(mid_chs, out_chs, kernel_size=1, apply_act=False, **ckwargs)
        self.drop_path = drop_path
        self.act3 = act_layer(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        # FIXME partial shortcut needed if first block handled as per original, not used for my current impl
        #x[:, :shortcut.size(1)] += shortcut
        x = self.act3(x)
        return x


class DarkBlock(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None, 
                 aa_layer=None, drop_block=None, drop_path=None):
        super(DarkBlock, self).__init__()
        norm_kwargs = norm_kwargs or {}
        mid_chs = int(round(out_chs * bottle_ratio))

        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, norm_kwargs=norm_kwargs, 
            aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, out_chs, kernel_size=3, dilation=dilation, 
            groups=groups, **ckwargs)
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        return x


class CrossStage(nn.Module):
    """Cross Stage."""
    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1., bottle_ratio=1., exp_ratio=1.,
                 groups=1, first_dilation=None, down_growth=False, cross_linear=False, block_dpr=None,
                 block_fn=ResBottleneck, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs  # grow downsample channels to output channels
        exp_chs = int(round(out_chs * exp_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'),
            norm_kwargs=block_kwargs.get('norm_kwargs'))

        if stride != 1 or first_dilation != dilation:
            self.conv_down = ConvBnAct(in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, 
                groups=groups, aa_layer=block_kwargs.get('aa_layer', None), **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs

        # FIXME this 1x1 expansion is pushed down into the cross and block paths in the darknet cfgs. Also,
        # there is also special case for the first stage for some of the model that results in uneven split
        # across the two paths. I did it this way for simplicity for now.
        self.conv_exp = ConvBnAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2  # output of conv_exp is always split in two

        self.blocks = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(block_dpr[i]) if block_dpr and block_dpr[i] else None
            self.blocks.add_module(str(i), block_fn(prev_chs, block_out_chs, dilation, 
                bottle_ratio, groups, drop_path=drop_path, **block_kwargs))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition_b = ConvBnAct(prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvBnAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        x = self.conv_exp(x)
        xs, xb = x.chunk(2, dim=1)
        xb = self.blocks(xb)
        out = self.conv_transition(torch.cat([xs, self.conv_transition_b(xb)], dim=1))
        return out


    

class CSPNet(nn.Module):
    """Cross Stage Partial base model.

    Paper: `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
    Ref Impl: https://github.com/WongKinYiu/CrossStagePartialNetworks

    NOTE: There are differences in the way I handle the 1x1 'expansion' conv in this impl vs the
    darknet impl. I did it this way for simplicity and less special cases.
    """

    def __init__(self, cfg, in_channel=3, num_classes=1000, output_stride=32, drop_rate=0.,
                 act_layer=nn.LeakyReLU, block_fn=ResBottleneck, aa_layer=None, drop_path_rate=0.,
                 zero_init_last_bn=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer, 
            norm_kwargs=norm_kwargs, aa_layer=aa_layer)

        # Construct the stem
        self.stem, stem_feat_info = create_stem(in_channel, **cfg['stem'], **layer_args)
        prev_chs = stem_feat_info['num_chs']
        curr_stride = stem_feat_info['reduction']  # reduction does not include pool
        if cfg['stem']['pool']:
            curr_stride *= 2

        self.out_channels = [stem_feat_info['num_chs']] + list(cfg['stage']['out_chs'])

        # Construct the stages
        per_stage_args = self._cfg_to_stage_args(cfg['stage'], curr_stride=curr_stride, 
            output_stride=output_stride, drop_path_rate=drop_path_rate)
        self.stages = nn.Sequential()
        for i, sa in enumerate(per_stage_args):
            self.stages.add_module(str(i), CrossStage(prev_chs, **sa, **layer_args, block_fn=block_fn))
            prev_chs = sa['out_chs']
            curr_stride *= sa['stride']

        # Construct the head
        self.num_features = prev_chs
        self.head = ClassifierHead(in_chs=prev_chs, num_classes=num_classes, drop_rate=drop_rate)
        self.out_channels.append(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _cfg_to_stage_args(self, cfg, curr_stride=2, output_stride=32, drop_path_rate=0.):
        # get per stage args for stage and containing blocks, calculate strides to meet target output_stride
        num_stages = len(cfg['depth'])
        if 'groups' not in cfg:
            cfg['groups'] = (1,) * num_stages
        if 'down_growth' in cfg and not isinstance(cfg['down_growth'], (list, tuple)):
            cfg['down_growth'] = (cfg['down_growth'],) * num_stages
        if 'cross_linear' in cfg and not isinstance(cfg['cross_linear'], (list, tuple)):
            cfg['cross_linear'] = (cfg['cross_linear'],) * num_stages
        cfg['block_dpr'] = [None] * num_stages if not drop_path_rate else \
            [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg['depth'])).split(cfg['depth'])]
        stage_strides = []
        stage_dilations = []
        stage_first_dilations = []
        dilation = 1
        for cfg_stride in cfg['stride']:
            stage_first_dilations.append(dilation)
            if curr_stride >= output_stride:
                dilation *= cfg_stride
                stride = 1
            else:
                stride = cfg_stride
                curr_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        cfg['stride'] = stage_strides
        cfg['dilation'] = stage_dilations
        cfg['first_dilation'] = stage_first_dilations
        stage_args = [dict(zip(cfg.keys(), values)) for values in zip(*cfg.values())]
        return stage_args

    def get_stages(self):
        out_channels, first_stage = None, None
        start_stages = 0
        if len(self.out_channels) == 6:
            out_channels = self.out_channels[:-1]
            first_stage = self.stem
        elif len(self.out_channels) == 7:
            out_channels = self.out_channels[1:-1]
            first_stage = nn.Sequential(self.stem, self.stages[0])
            start_stages = 1
        else:
            raise RuntimeError("Unknown error, report this as a bug!")

        stages = [first_stage] + [m for m in self.stages[start_stages:]]
        return nn.Sequential(*stages), out_channels

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.head = ClassifierHead(self.num_features, num_classes, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _cspnet(arch, pretrained, progress, **kwargs):
    num_classes = 1000
    if pretrained and kwargs.get("num_classes", False):
        num_classes = kwargs.pop("num_classes")

    model = CSPNet(model_cfgs[arch], **kwargs)
    if pretrained:
        load_pretrained(model, model_urls[arch], num_classes=num_classes, 
            first_conv_name="stem.conv1", classifier_name="head.fc", progress=progress)
    return model


def cspresnet50(pretrained=False, progress=True, **kwargs):
    return _cspnet('cspresnet50', pretrained=pretrained, progress=progress, 
        **kwargs)

def cspresnext50(pretrained=False, progress=True, **kwargs):
    return _cspnet('cspresnext50', pretrained=pretrained, progress=progress, 
        **kwargs)

def cspdarknet53(pretrained=False, progress=True, **kwargs):
    return _cspnet('cspdarknet53', pretrained=pretrained, progress=progress, 
        block_fn=DarkBlock, **kwargs)


def get_backbone(model_name: str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, **kwargs):
    if not model_name in supported_models:
        raise RuntimeError("model {} is not supported yet, available: {}".format(model_name, supported_models))

    network = eval('{}(pretrained=pretrained, num_classes=n_classes, **kwargs)'.format(model_name))
    stages, channels = network.get_stages()

    if feature_type == "tri_stage_fpn":
        backbone = Backbone(stages, channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(stages, network.get_classifier(), n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got {}".format(feature_type))
    return backbone
