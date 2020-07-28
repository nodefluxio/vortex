import vortex
from . import modules

def test_alexnet():
    assert 'alexnet' in vortex.networks.models.all_models
    assert vortex.networks.models.remove_model('alexnet')
    assert 'alexnet' not in vortex.networks.models.all_models

def test_squeezenet():
    assert 'squeezenetv1.0' in vortex.networks.modules.backbones.all_models
    assert 'squeezenetv1.1' in vortex.networks.modules.backbones.all_models
    assert vortex.networks.modules.backbones.remove_backbone('squeezenetv1.0')
    assert 'squeezenetv1.0' not in vortex.networks.modules.backbones.all_models
    assert 'squeezenetv1.1' not in vortex.networks.modules.backbones.all_models