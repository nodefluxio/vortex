import torch.nn as nn

from .base_backbone import BackboneConfig, BackboneBase
from ..utils.arch_utils import load_pretrained


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

supported_models = [
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
]

_complete_url = lambda x: 'https://download.pytorch.org/models/' + x
default_cfgs = {
    'vgg11': BackboneConfig(pretrained_url=_complete_url('vgg11-bbd30ac9.pth')),
    'vgg13': BackboneConfig(pretrained_url=_complete_url('vgg13-c768596a.pth')),
    'vgg16': BackboneConfig(pretrained_url=_complete_url('vgg16-397923af.pth')),
    'vgg19': BackboneConfig(pretrained_url=_complete_url('vgg19-dcbb9e9d.pth')),
    'vgg11_bn': BackboneConfig(pretrained_url=_complete_url('vgg11_bn-6002323d.pth')),
    'vgg13_bn': BackboneConfig(pretrained_url=_complete_url('vgg13_bn-abd245e5.pth')),
    'vgg16_bn': BackboneConfig(pretrained_url=_complete_url('vgg16_bn-6c64b313.pth')),
    'vgg19_bn': BackboneConfig(pretrained_url=_complete_url('vgg19_bn-c79401a0.pth')),
}

class VGG(BackboneBase):

    def __init__(self, name, features, num_classes=1000, init_weights=True, norm_layer=None, norm_kwargs=None, default_config=None):
        super(VGG, self).__init__(name, default_config)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            norm_kwargs = {}

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._num_classes = num_classes

        self._stages_channel = (64, 128, 256, 512, 512)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def get_stages(self):
        stages, tmp = [], []
        for m in self.features:
            tmp.append(m)
            if isinstance(m, nn.MaxPool2d):
                stages.append(nn.Sequential(*tmp))
                tmp = []
        return nn.Sequential(*stages)

    @property
    def stages_channel(self):
        return self._stages_channel

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_classifer_feature(self):
        return 4096

    def get_classifier(self) :
        return nn.Sequential(
            self.avgpool,
            self.flatten,
            self.classifier
        )

    def reset_classifier(self, num_classes, classifier=None):
        self._num_classes = num_classes
        if num_classes < 0:
            classifier = nn.Identity()
        elif classifier is None:
            classifier = nn.Linear(4096, num_classes)
        if not isinstance(classifier, nn.Module):
            raise TypeError("'classifier' argument is required to have type of 'int' or 'nn.Module', "
                "got {}".format(type(classifier)))
        self.classifier[-1] = classifier

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, norm_layer=None, norm_kwargs=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    if norm_kwargs is None:
        norm_kwargs = {}

    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, norm_layer(v, **norm_kwargs), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



model_cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, batch_norm, pretrained, progress, **kwargs):
    num_classes = 1000
    if pretrained:
        kwargs['init_weights'] = False
        if "num_classes" in kwargs:
            num_classes = kwargs.pop("num_classes")

    arch_stripped = arch.split('_')[0]
    norm_layer, norm_kwargs = None, None
    if 'norm_layer' in kwargs:
        norm_layer = kwargs['norm_layer']
    if 'norm_kwargs' in kwargs:
        norm_kwargs = kwargs['norm_kwargs']
    features = make_layers(model_cfgs[arch_stripped], batch_norm=batch_norm, 
        norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    model = VGG(arch, features, default_config=default_cfgs[arch], **kwargs)
    if pretrained:
        load_pretrained(model, default_cfgs[arch].pretrained_url, num_classes=num_classes, 
            first_conv_name="features.0", progress=progress)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', True, pretrained, progress, **kwargs)
