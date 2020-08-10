import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    'round_channels'
]


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)


def load_pretrained(model, url, in_channel=3, num_classes=1000, first_conv_name=None, 
                    classifier_name=None, progress=True):
    assert in_channel in (1, 3)

    state_dict = load_state_dict_from_url(url, progress=progress)
    if in_channel != 3:
        if first_conv_name is None:
            raise RuntimeError("could not change the default number of input channel "\
                "without 'first_conv_name' argument provided")
        first_conv_name += ".weight"
        state_dict[first_conv_name] = state_dict[first_conv_name].sum(dim=1, keepdim=True)   
    model.load_state_dict(state_dict, strict=True)

    if num_classes != 1000:
        if getattr(model, 'reset_classifier', None):
            model.reset_classifier(num_classes)
        elif classifier_name:
            if not getattr(model, classifier_name, None):
                raise RuntimeError("'classifier_name' of {} is not available in the "\
                    "attribute of the model provided".format(classifier_name))
            in_features = getattr(model, classifier_name).in_features
            classifier = nn.Linear(in_features, num_classes)
            setattr(model, classifier_name, classifier)
        else:
            raise RuntimeError("could not change the default number of class for classifier "\
                "without 'classifier_name' argument or 'reset_classifier' attribute in model provided")
