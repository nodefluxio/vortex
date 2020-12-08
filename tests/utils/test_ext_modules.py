import vortex.development

import examples.custom_models.alexnet.alexnet as alexnet
import examples.custom_backbones.squeezenet.squeezenet as squeezenet


def test_alexnet():
    assert 'alexnet' in vortex.development.networks.models.all_models
    assert vortex.development.networks.models.remove_model('alexnet')
    assert 'alexnet' not in vortex.development.networks.models.all_models

def test_squeezenet():
    assert 'squeezenetv1.0' in vortex.development.networks.modules.backbones.all_models
    assert 'squeezenetv1.1' in vortex.development.networks.modules.backbones.all_models
    assert vortex.development.networks.modules.backbones.remove_backbone('squeezenetv1.0')
    assert 'squeezenetv1.0' not in vortex.development.networks.modules.backbones.all_models
    assert 'squeezenetv1.1' not in vortex.development.networks.modules.backbones.all_models
