"""
    MobileNetV3 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
"""

supported_models = [
    'mobilenetv3_small_w7d20', 'mobilenetv3_small_wd2', 'mobilenetv3_small_w3d4',
    'mobilenetv3_small_w1', 'mobilenetv3_small_w5d4', 'mobilenetv3_large_w7d20', 'mobilenetv3_large_wd2',
    'mobilenetv3_large_w3d4', 'mobilenetv3_large_w1', 'mobilenetv3_large_w5d4'
]

__all__ = ['MobileNetV3']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import logging
import hashlib
import zipfile
import warnings

from .base_backbone import Backbone, ClassifierFeature
from vortex.development.networks.modules.utils.conv2d import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block
from vortex.development.networks.modules.utils.arch_utils import round_channels
from vortex.development.networks.modules.utils.activations import hard_sigmoid, hard_swish, sigmoid, swish


_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('mobilenetv3_large_w1', '0779', '38e392f58bdf99b2832b26341bc9704ac63a3672', 'v0.0.411'),
]}


imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=torch.sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x : torch.Tensor):
        # NOTE adaptiveavgpool bad for NVIDIA AMP performance
        # tensor.view + mean bad for ONNX export (produces mess of gather ops that break TensorRT)
        #x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class MobileNetV3Unit(nn.Module):
    """
    MobileNetV3 unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    exp_channels : int
        Number of middle (expanded) channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    activation : str
        Activation function or name of activation function.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_channels,
                 stride,
                 use_kernel3,
                 activation,
                 use_se):
        super(MobileNetV3Unit, self).__init__()
        assert (exp_channels >= out_channels)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se
        self.use_exp_conv = exp_channels != out_channels
        mid_channels = exp_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation)
        else :
            self.exp_conv = nn.Identity()
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        if self.use_se:
            self.se = SqueezeExcite(
                in_chs=mid_channels,
                reduce_chs=4
            )
            # self.se = SEBlock(
            #     channels=mid_channels,
            #     reduction=4,
            #     approx_sigmoid=True,
            #     round_mid=True)
        else:
            self.se = nn.Identity()
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        else:
            identity = torch.zeros_like(x)
        # if self.use_exp_conv:
            # x = self.exp_conv(x)
        x = self.exp_conv(x)
        x = self.conv1(x)
        # if self.use_se:
            # x = self.se(x)
        x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV3FinalBlock(nn.Module):
    """
    MobileNetV3 final block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_se):
        super(MobileNetV3FinalBlock, self).__init__()
        self.use_se = use_se

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=hard_swish
            # activation="hswish"
        )
        if self.use_se:
            self.se = SqueezeExcite(
                in_chs=out_channels,
                reduce_chs=4
            )
            # self.se = SEBlock(
            #     channels=out_channels,
            #     reduction=4,
            #     approx_sigmoid=True,
            #     round_mid=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3 classifier.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 dropout_rate):
        super(MobileNetV3Classifier, self).__init__()
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        # self.activ = HSwish(inplace=True)
        self.activ = hard_swish
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.conv1(x)
        x = self.activ(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x


class MobileNetV3(nn.Module):
    """
    MobileNetV3 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    exp_channels : list of list of int
        Number of middle (expanded) channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    use_relu : list of list of int/bool
        Using ReLU activation flag for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
    final_use_se : bool
        Whether to use SE-module in the final block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 exp_channels,
                 init_block_channels,
                 final_block_channels,
                 classifier_mid_channels,
                 kernels3,
                 use_relu,
                 use_se,
                 first_stride,
                 final_use_se,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            activation=hard_swish
            # activation="hswish"
        ))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                exp_channels_ij = exp_channels[i][j]
                stride = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                use_kernel3 = kernels3[i][j] == 1
                # activation = "relu" if use_relu[i][j] == 1 else "hswish"
                activation = F.relu if use_relu[i][j] else hard_swish
                use_se_flag = use_se[i][j] == 1
                stage.add_module("unit{}".format(j + 1), MobileNetV3Unit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    exp_channels=exp_channels_ij,
                    use_kernel3=use_kernel3,
                    stride=stride,
                    activation=activation,
                    use_se=use_se_flag))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', MobileNetV3FinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            use_se=final_use_se))
        in_channels = final_block_channels

        self.output = MobileNetV3Classifier(
            in_channels=in_channels,
            out_channels=num_classes,
            mid_channels=classifier_mid_channels,
            dropout_rate=0.2)

        self._init_params()

    @torch.jit.ignore
    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x
    
    def reset_classifier(self, num_classes):
        in_channel = self.output.conv2.in_channels
        self.output.conv2 = conv1x1(in_channel, num_classes, bias=True)


## TODO : move to utility
def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


## TODO : move to utility
def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".torch", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.
    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.pth".format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning("Mismatch in the content of model file detected. Downloading again.")
    else:
        logging.info("Model file not found. Downloading to {}.".format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + ".zip"
    _download(
        url="{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip".format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError("Downloaded file has different hash. Please try again.")


## TODO : move to utility
def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """
    Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. " \
            "Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning("File {} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the `repo_url` is overridden, consider switching to "
                                      "the default repo.".format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, "s" if retries > 1 else ""))

    return fname


## TODO : move to utility
def _check_sha1(file_name, sha1_hash):
    """
    Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    file_name : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(file_name, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


## TODO : move to utility
def load_model(net,
               file_path,
               ignore_extra=True,
               strict=False):
    """
    Load model state dictionary from a file.
    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state,strict=strict)
    else:
        net.load_state_dict(torch.load(file_path),strict=strict)

## TODO : move to utility
def download_model(net,
                   model_name,
                   local_model_store_dir_path=os.path.join("~", ".torch", "models"),
                   ignore_extra=True):
    """
    Load model state dictionary from a file with downloading it if necessary.
    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    load_model(
        net=net,
        file_path=get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path),
        ignore_extra=ignore_extra)


def get_mobilenetv3(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".torch", "models"),
                    **kwargs):
    """
    Create MobileNetV3 model with specific parameters.
    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('small' or 'large').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if version == "small":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        first_stride = True
        final_block_channels = 576
    elif version == "large":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        first_stride = False
        final_block_channels = 960
    else:
        raise ValueError("Unsupported MobileNetV3 version {}".format(version))

    final_use_se = False
    classifier_mid_channels = 1280

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        exp_channels = [[round_channels(cij * width_scale) for cij in ci] for ci in exp_channels]
        init_block_channels = round_channels(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale)

    num_classes = 1000
    if pretrained and "num_classes" in kwargs:
        num_classes = kwargs.pop("num_classes")

    net = MobileNetV3(
        channels=channels,
        exp_channels=exp_channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        use_relu=use_relu,
        use_se=use_se,
        first_stride=first_stride,
        final_use_se=final_use_se,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)
        if num_classes != 1000:
            net.reset_classifier(num_classes)
    return net


def mobilenetv3_small_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.35, 
        model_name="mobilenetv3_small_w7d20", **kwargs), [8, 8, 8, 16, 576]


def mobilenetv3_small_wd2(**kwargs):
    """
    MobileNetV3 Small 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.5, 
        model_name="mobilenetv3_small_wd2", **kwargs), [8, 8, 16, 24, 576]


def mobilenetv3_small_w3d4(**kwargs):
    """
    MobileNetV3 Small 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.75, 
        model_name="mobilenetv3_small_w3d4", **kwargs), [16, 16, 24, 40, 576]


def mobilenetv3_small_w1(**kwargs):
    """
    MobileNetV3 Small 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.0, 
        model_name="mobilenetv3_small_w1", **kwargs), [16, 16, 24, 48, 576]


def mobilenetv3_small_w5d4(**kwargs):
    """
    MobileNetV3 Small 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.25, 
        model_name="mobilenetv3_small_w5d4", **kwargs), [24, 24, 32, 64, 720]


def mobilenetv3_large_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.35, 
        model_name="mobilenetv3_small_w7d20", **kwargs), [8, 8, 16, 40, 960]


def mobilenetv3_large_wd2(**kwargs):
    """
    MobileNetV3 Large 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.5, 
        model_name="mobilenetv3_large_wd2", **kwargs), [8, 16, 24, 56, 960]


def mobilenetv3_large_w3d4(**kwargs):
    """
    MobileNetV3 Large 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.75, 
        model_name="mobilenetv3_large_w3d4", **kwargs), [16, 24, 32, 88, 960]


def mobilenetv3_large_w1(**kwargs):
    """
    MobileNetV3 Large 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.0, 
        model_name="mobilenetv3_large_w1", **kwargs), [16, 24, 40, 112, 960]


def mobilenetv3_large_w5d4(**kwargs):
    """
    MobileNetV3 Large 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.25, 
        model_name="mobilenetv3_large_w5d4", **kwargs), [24, 32, 48, 144, 1200]

def _mobilenetv3_small_stages(network : MobileNetV3) :
    return nn.Sequential(
        network.features.init_block,
        network.features.stage1,
        network.features.stage2,
        network.features.stage3,
        nn.Sequential(
            network.features.stage4,
            network.features.final_block
        )
    ) ## note : 576 includes final block else 96

def _mobilenetv3_large_stages(network : MobileNetV3) :
    return nn.Sequential(
        nn.Sequential(
            network.features.init_block,
            network.features.stage1,
        ),
        network.features.stage2,
        network.features.stage3,
        network.features.stage4,
        nn.Sequential(
            network.features.stage5,
            network.features.final_block,
        )
    ) ## note : 960 includes final block else 160

def get_backbone(model_name : str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, *args, **kwargs):
    if not model_name in supported_models:
        raise RuntimeError("model %s is not supported yet, available : %s" %(model_name, supported_models))
    if len(args) != 0:
        warnings.warn("unused argument(s) in 'get_backbone': %s" % args)
    network, channels = eval('{}(pretrained=pretrained, num_classes=n_classes, **kwargs)'.format(model_name))
    stages =  _mobilenetv3_small_stages(network) if 'small' in model_name \
        else _mobilenetv3_large_stages(network)
    
    if feature_type == "tri_stage_fpn":
        backbone = Backbone(stages, channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(stages, network.output, n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got %s" % (feature_type))
    return backbone
