import vortex
from . import modules

def test_alexnet():
    assert 'alexnet' in vortex.networks.models.all_models

def test_squeezenet():
    assert 'squeezenetv1.0' in vortex.networks.modules.backbones.all_models
    assert 'squeezenetv1.1' in vortex.networks.modules.backbones.all_models