import yaml
import enforce
import easydict
import copy

from .loader import Loader

from enum import Enum, unique
from pathlib import Path
from collections import namedtuple
from typing import Union, Dict, Any, List, Tuple
from anytree import Node, RenderTree, NodeMixin
from anytree.exporter import UniqueDotExporter


@unique
class ExperimentType(Enum):
    UNKNOWN = 'unknown'
    HYPOPT = 'hypopt'
    TRAIN = 'train'
    VALIDATE = 'validate'
    EXPORT = 'export'


class ExperimentNode(NodeMixin):
    def __init__(self, name, parent=None, children=None, required: bool = True, docstring: str = '', **kwargs):
        self.separator = '.'
        self.required = required
        super(ExperimentNode, self).__init__()
        self.name = name
        self.parent = parent
        self.docstring = docstring
        if children:
            self.children = children

    def is_required(self):
        return all([node.required for node in self.path])

    def is_optional(self):
        return any([not node.required for node in self.path])

    def __str__(self):
        path = self.path[1:] if self.parent else self.path
        # join path, then remove first separator
        return ("%s" % self.separator.join([""] + [str(node.name) for node in path]))[1:]

    def __repr__(self):
        args = [str(self)]
        classname = self.__class__.__name__
        nameblacklist = ["name", "separator"]
        for key, value in filter(lambda item: not item[0].startswith("_") and item[0] not in nameblacklist,
                                 sorted(self.__dict__.items(),
                                        key=lambda item: item[0])):
            args.append("%s=%r" % (key, value))
        return "%s(%s)" % (classname, ", ".join(args))


def __logging_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    logging = ExperimentNode('logging', required=required,
                             docstring='comet_ml logging config')
    module = ExperimentNode('module', parent=logging, required=False)
    module_args = ExperimentNode('args', parent=logging, required=False)
    pytz_timezone = ExperimentNode('pytz_timezone', parent=logging, required=False)
    # api_key = ExperimentNode('api_key', parent=logging)
    # project_name = ExperimentNode('project_name', parent=logging)
    # workspace = ExperimentNode('workspace', parent=logging)
    return logging


def __model_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    model = ExperimentNode('model', required=required,
                           docstring='model and loss definition')
    model_name = ExperimentNode('name', parent=model)
    network_args = ExperimentNode('network_args', parent=model)
    preprocess_args = ExperimentNode('preprocess_args', parent=model)
    input_size = ExperimentNode('input_size', parent=preprocess_args)
    input_normalization = ExperimentNode(
        'input_normalization', parent=preprocess_args)
    pixel_mean = ExperimentNode('mean', parent=input_normalization)
    pixel_std = ExperimentNode('std', parent=input_normalization)
    
    if (exp_step == ExperimentType.TRAIN) or (exp_step == ExperimentType.HYPOPT):
        # loss = ExperimentNode('loss', parent=model)
        loss_args = ExperimentNode('loss_args', parent=model)
        init_state_dict = ExperimentNode(
            'init_state_dict', parent=model, required=False)
        init_file = ExperimentNode('file', parent=init_state_dict)
        skip_ok = ExperimentNode('skip_ok', parent=init_state_dict)
        args = ExperimentNode('args', parent=init_state_dict,
                              docstring='forwarded to torch.load')
    elif (exp_step == ExperimentType.EXPORT) or (exp_step == ExperimentType.VALIDATE):
        postprocess_args = ExperimentNode('postprocess_args', parent=model, required=False)
    return model


def __seed_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    seed = ExperimentNode('seed', required=required,
                          docstring='seed for random sample')
    torch_ = ExperimentNode('torch', parent=seed, required=False)
    numpy_ = ExperimentNode('numpy', parent=seed, required=False)
    random_ = ExperimentNode('random', parent=seed, required=False)
    cudnn = ExperimentNode('cudnn', parent=seed, required=False)
    benchmark = ExperimentNode('benchmark', parent=cudnn, required=False)
    deterministic = ExperimentNode(
        'deterministic', parent=cudnn, required=False)
    return seed


# def __preprocess_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
#     preprocess = ExperimentNode(
#         'preprocess', required=required, docstring='prepocess module')
#     method = ExperimentNode('method', parent=preprocess)
#     args = ExperimentNode('args', parent=preprocess)
#     return preprocess


# def __postprocess_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
#     postprocess = ExperimentNode(
#         'postprocess', required=required, docstring='postprocess module')
#     method = ExperimentNode('method', parent=postprocess)
#     args = ExperimentNode('args', parent=postprocess)
#     return postprocess


def __dataloader_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    dataloader = ExperimentNode('dataloader', required=required,
                                docstring='dataloader currently only support pytorch DataLoader')
    loader = ExperimentNode('module', parent=dataloader)
    loader_args = ExperimentNode('args', parent=dataloader)
    # collater = ExperimentNode('collater', parent=dataloader)
    # collater_class = ExperimentNode('collater', parent=collater)
    # collater_args = ExperimentNode('args', parent=collater)
    return dataloader


def __validation_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    validation = ExperimentNode(
        'validation', required=required, docstring='validator config')
    validation_args = ExperimentNode('args', parent=validation)
    validation_score = ExperimentNode('score_threshold', parent=validation_args, required=False)
    validation_iou = ExperimentNode('iou_threshold', parent=validation_args, required=False)
    return validation

def __validator_tree(required=True, parent=None):
    validator = ExperimentNode('validator', required=required, parent=parent, docstring='validator config')
    pred_args = ExperimentNode('predictor_args', parent=validator, required=False)
    addition_args = ExperimentNode('additional_args', parent=validator, required=False)
    val_epoch = ExperimentNode('val_epoch', parent=validator, required=False)
    return validator


def __optimizer_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    optimizer = ExperimentNode(
        'optimizer', required=required, docstring='optimizer config')
    optimizer_method = ExperimentNode('method', parent=optimizer)
    optimizer_args = ExperimentNode('args', parent=optimizer)
    return optimizer


def __exporter_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    exporter = ExperimentNode('exporter', required=required)
    size = ExperimentNode('size', parent=exporter)
    output_name = ExperimentNode('output_name', parent=exporter)
    input_name = ExperimentNode('input_name', parent=exporter)
    input_dtype = ExperimentNode('input_dtype', parent=exporter)
    opset_version = ExperimentNode('opset_version', parent=exporter)
    args = ExperimentNode('args', parent=exporter)
    graph_ops = ExperimentNode('graph_ops', parent=exporter)
    state_dict = ExperimentNode('state_dict', parent=exporter)
    return exporter


def __scheduler_tree(required: bool = True, parent=None, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    ## only for backward compatibility
    scheduler = ExperimentNode('scheduler', required=False, parent=parent, 
        docstring='learning rate scheduler config')
    scheduler_method = ExperimentNode('method', parent=scheduler)
    scheduler_args = ExperimentNode('args', parent=scheduler)

    lr_scheduler = ExperimentNode('lr_scheduler', required=required, parent=parent, 
        docstring='learning rate scheduler config')
    lr_scheduler_method = ExperimentNode('method', parent=lr_scheduler)
    lr_scheduler_args = ExperimentNode('args', parent=lr_scheduler)
    return lr_scheduler, scheduler


def __dataset_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    dataset = ExperimentNode(
        'dataset', required=required, docstring='dataset confg')
    if (exp_step == ExperimentType.TRAIN) or (exp_step == ExperimentType.HYPOPT):
        train_required, add_train = True, True
        eval_required, add_eval = False, True
    elif exp_step == ExperimentType.VALIDATE:
        eval_required, add_eval = True, True
        train_required, add_train = False, False
    else:
        train_required, add_train = False, False
        eval_required, add_eval = False, False
    if add_train:
        dataset_train = ExperimentNode('train', parent=dataset, required=train_required, 
            docstring='training dataset')
        dataset_train_cls = ExperimentNode('dataset', parent=dataset_train, required=False, 
            docstring='dataset class')
        dataset_train_name = ExperimentNode('name', parent=dataset_train, required=train_required,
            docstring='train dataset name')
        dataset_train_args = ExperimentNode('args', parent=dataset_train, required=train_required, 
            docstring='arguments to be passed to dataset class')
        dataset_train_augs = ExperimentNode('augmentations', parent=dataset_train, required=False, 
            docstring='augmentations for train dataset')
    if add_eval:
        dataset_eval = ExperimentNode('eval', parent=dataset, required=eval_required, 
            docstring='validation dataset')
        dataset_eval_cls = ExperimentNode('dataset', parent=dataset_eval, required=False, 
            docstring='dataset class')
        dataset_eval_name = ExperimentNode('name', parent=dataset_eval, required=eval_required,
            docstring='validation dataset name')
        dataset_eval_args = ExperimentNode('args', parent=dataset_eval, required=eval_required, 
            docstring='arguments to be passed to dataset class')
    return dataset


def __trainer_tree(required: bool = True, exp_step: ExperimentType = ExperimentType.UNKNOWN):
    trainer = ExperimentNode('trainer', docstring='trainer configuration')
    if (exp_step == ExperimentType.VALIDATE) or (exp_step == ExperimentType.HYPOPT):
        validation = __validation_tree(required=False)
        validation.parent = trainer
    elif exp_step == ExperimentType.TRAIN:
        validation = __validation_tree(required=False)
        validation.parent = trainer
        save_epoch = ExperimentNode('save_epoch', parent=trainer, required=False,
            docstring='every `save_epoch` epoch step, we will save the weights')
        save_best = ExperimentNode('save_best_metric', parent=trainer, required=False,
            docstring='metric name to save the best model')
    device = ExperimentNode('device', parent=trainer, required=False,
                            docstring='on which device we should train?')
    if (exp_step == ExperimentType.TRAIN) or (exp_step == ExperimentType.HYPOPT):
        optimizer = __optimizer_tree()
        optimizer.parent = trainer
        lr_scheduler, scheduler = __scheduler_tree(required=False, parent=trainer)
        driver = ExperimentNode('driver', parent=trainer, required=True)
        driver_module = ExperimentNode('module', parent=driver, required=True)
        driver_args = ExperimentNode('args', parent=driver, required=False)
        epoch = ExperimentNode('epoch', parent=trainer,
                               docstring='number of training epoch')
    return trainer


def __train_config():
    exp_step = ExperimentType.TRAIN
    config = ExperimentNode(
        'config', docstring='this is your experiment file, in your file')
    trainer = __trainer_tree(required=True, exp_step=exp_step)
    trainer.parent = config
    model = __model_tree(required=True, exp_step=exp_step)
    model.parent = config
    logging = __logging_tree(required=True)
    logging.parent = config
    dataset = __dataset_tree(required=True, exp_step=exp_step)
    dataset.parent = config
    dataloader = __dataloader_tree(required=True, exp_step=exp_step)
    dataloader.parent = config
    validator = __validator_tree(required=False, parent=config)
    seed = __seed_tree(False, exp_step=exp_step)
    seed.parent = config
    name = ExperimentNode(
        'experiment_name', docstring='this is your experiment name', parent=config)
    output_directory = ExperimentNode('output_directory', required=False,
                                      docstring='where should we put your experiment output(s)?', parent=config)
    return config


def __hypopt_config():
    exp_step = ExperimentType.TRAIN
    config = ExperimentNode('config', docstring='this is your experiment file')
    parameters = ExperimentNode(
        'parameters', docstring='list of dict describing which parameter to optimize, key : parameter<str>; value : config<dict : suggestion & args>', parent=config)
    override = ExperimentNode(
        'override', docstring='dict describing which config parameter to override', parent=config)
    additional_config = ExperimentNode(
        'additional_config', docstring='dict describing parameters to be added to config (mainly for training)', parent=config)
    study = ExperimentNode(
        'study', docstring='configuration for optuna study', parent=config)
    direction = ExperimentNode(
        'direction', docstring='optimization direction [maximize, minimize]', parent=study)
    study_name = ExperimentNode(
        'study_name', docstring='well, study name', parent=study)
    args = ExperimentNode(
        'args', docstring='additional args that will be forwarded to optuna', parent=study)
    logging = __logging_tree(required=True)
    logging.parent = config
    name = ExperimentNode(
        'experiment_name', docstring='this is your experiment name', parent=config)
    return config


def __validate_config():
    exp_step = ExperimentType.VALIDATE
    config = ExperimentNode('config', docstring='this is your experiment file')
    trainer = __trainer_tree(required=True, exp_step=exp_step)
    trainer.parent = config
    model = __model_tree(required=True, exp_step=exp_step)
    model.parent = config
    dataset = __dataset_tree(required=True, exp_step=exp_step)
    dataset.parent = config
    dataloader = __dataloader_tree(required=False, exp_step=exp_step)
    dataloader.parent = config
    validator = __validator_tree(required=True, parent=config)
    name = ExperimentNode(
        'experiment_name', docstring='this is your experiment name', parent=config)
    output_directory = ExperimentNode(
        'output_directory', docstring='where should we put your experiment output(s)?', parent=config, required=False)
    return config


def __export_config():
    exp_step = ExperimentType.EXPORT
    config = ExperimentNode('config', docstring='this is your experiment file')
    name = ExperimentNode(
        'name', docstring='this is your experiment name', parent=config)
    output_directory = ExperimentNode(
        'output_directory', docstring='where should we put your experiment output(s)?', parent=config, required=False)
    exporter = __exporter_tree()
    exporter.parent = config
    model = __exporter_tree()
    model.parent = config
    return config


def __get_config(experiment_type: str):
    supported_exp = any([(experiment_type == enum.value)
                         for enum in ExperimentType])
    if not supported_exp:
        raise RuntimeError('unsupported experiment : %s' % experiment_type)
    exp_req = None
    if experiment_type == 'train':
        exp_req = __train_config()
    elif experiment_type == 'validate':
        exp_req = __validate_config()
    elif experiment_type == 'hypopt':
        exp_req = __hypopt_config()
    elif experiment_type == 'export':
        exp_req = __export_config()

    if experiment_type in ('train', 'validate', 'hypopt'):
        ckpt = ExperimentNode('checkpoint', parent=exp_req, required=False)
    return exp_req


def show_config_requirements(experiment_type: str, format: str = 'text'):
    """
    show experiment file requirements
    Arguments:
        - experiment_type : str
        - format : str
    """
    if not format in ['text', 'figure']:
        raise RuntimeError(
            'unsupported format %s, supported : `text`, `figure`' % (format))
    config = __get_config(experiment_type)
    if format == 'text':
        return str(RenderTree(config))
    else:
        filename = 'config_requirements.gv'
        from tempfile import NamedTemporaryFile
        try:
            from graphviz import Source
        except ImportError as e:
            raise ImportError(
                "unable to import graphviz, you can install it using `pip3 install graphviz`")
        with NamedTemporaryFile("wb", delete=False) as dotfile:
            dotfilename = dotfile.name

            def edgetypefunc(node, child):
                return '->'

            def nodeattrfunc(node):
                return 'label="%s", %s' % (node.name, "shape=box" if node.is_required() else "shape=oval")
            exporter = UniqueDotExporter(
                config, edgetypefunc=edgetypefunc, nodeattrfunc=nodeattrfunc)
            exporter.to_dotfile('test.dot')
            for line in exporter:
                dotfile.write(("%s\n" % line).encode("utf-8"))
            dotfile.flush()
            gv = Source.from_file(dotfilename)
            return gv


def _check_deprecation(config):
    import warnings
    def _warn_print(message, category, filename, lineno, file=None, line=None):
        print("DeprecationWarning:", message)
    _orig_show_warning = warnings.showwarning
    warnings.showwarning = _warn_print

    if 'dataset' in config and 'train' in config.dataset and 'dataset' in config.dataset.train:
        warnings.warn("'config.dataset.train.dataset' is now changed to "
            "'config.dataset.train.name'")
        config.dataset.train.name = config.dataset.train.pop('dataset')
    if 'dataset' in config and 'eval' in config.dataset and 'dataset' in config.dataset.eval:
        warnings.warn("'config.dataset.eval.dataset' is now changed to "
            "'config.dataset.eval.name'",)
        config.dataset.eval.name = config.dataset.eval.pop('dataset')

    if 'trainer' in config and 'device' in config.trainer and 'device' not in config:
        warnings.warn("'config.trainer.device' is now moved to "
            "'config.device'")
        config.device = config.trainer.pop('device')

    if 'dataset' in config and 'dataloader' in config.dataset and 'dataloader' not in config:
        warnings.warn("'config.dataset.dataloader' is now moved to "
            "'config.dataloader'")
        config.dataloader = config.dataset.pop('dataloader')
    if 'dataloader' in config and 'dataloader' in config.dataloader:
        warnings.warn("'config.dataset.dataloader.dataloader' is now changed to "
            "'config.dataloader.module'")
        config.dataloader.module = config.dataloader.pop('dataloader')
    if 'dataloader' in config and config.dataloader.module.lower() == 'dataloader':
        warnings.warn("'DataLoader' module in 'config.dataloader.module' is now "
            "changed to 'PytorchDataLoader'")
        config.dataloader.module = "PytorchDataLoader"

    if 'trainer' in config and 'scheduler' in config.trainer:
        warnings.warn("'config.trainer.scheduler' is now changed to "
            "'config.trainer.lr_scheduler'")
        config.trainer.lr_scheduler = config.trainer.pop('scheduler')

    if 'trainer' in config and 'validation' in config.trainer:
        warnings.warn("'config.trainer.validation' is now changed to "
            "'config.validator'")
        config.validator = config.trainer.pop('validation')
    if 'validation' in config:
        warnings.warn("validation config is now changed to 'config.validator'")
        config.validator = config.pop('validation')
    warnings.showwarning = _orig_show_warning
    return config

def _check_none_str(config):
    assert isinstance(config, dict)
    for k, v in config.items():
        if isinstance(v, str) and v.lower() == 'none':
            config[k] = None
        elif isinstance(v, dict):
            config[k] = _check_none_str(v)
    return config


def check_config(config: Union[Path, dict, easydict.EasyDict], experiment_type: str):
    if isinstance(config, str):
        config = Path(config)
    if isinstance(config, Path):
        config = load_config(config)
    if isinstance(config, dict):
        config = easydict.EasyDict(config)
    if not isinstance(config, easydict.EasyDict):
        raise TypeError('expects config to be `dict` or `EasyDict`')
    config = _check_deprecation(copy.deepcopy(config))
    exp_tree = RenderTree(__get_config(experiment_type))
    # >= python3.6
    # requireds : List[Dict[str,bool]] = [{str(field) : __has_attr(config,str(field))} for _, _, field in exp_tree if field.is_required()]
    # optionals : List[Dict[str,bool]] = [{str(field) : __has_attr(config,str(field))} for _, _, field in exp_tree if field.is_optional()]
    requireds = [{str(field): __has_attr(config, str(field))} for _, _,
                 field in exp_tree if field.is_required() and field.parent]  # ignore root
    optionals = [{str(field): __has_attr(config, str(field))} for _, _,
                 field in exp_tree if field.is_optional() and field.parent]  # ignore root
    valid = all([exists for required in requireds for key,
                 exists in required.items()])
    missing = [key for required in requireds for key,
               exists in required.items() if not exists]
    # >= python3.6
    # optionals_set : List[str] = [key for optional in optionals for key, value in optional.items() if value]
    # optionals_not_set : List[str] = [key for optional in optionals for key, value in optional.items() if not value]
    optionals_set = [key for optional in optionals for key,
                     value in optional.items() if value]
    optionals_not_set = [key for optional in optionals for key,
                         value in optional.items() if not value]
    return ConfigCheckResult(valid, missing, optionals_set, optionals_not_set, experiment_type)


def load_config(filename: Union[Path, str]) -> easydict.EasyDict:
    file = Path(filename)
    if not file.exists():
        raise RuntimeError("file %s does not exists!" % str(file))
    with file.open() as f:
        config = yaml.load(f, Loader)
    config = easydict.EasyDict(config)
    config = _check_deprecation(config)
    config = _check_none_str(config)
    return config


def __has_attr(config: easydict.EasyDict, attr: str):
    ok = True
    try:
        # TODO : check for attribute without assignmetn
        _ = eval('config.%s' % attr)
    except AttributeError as e:
        ok = False
    return ok


class ConfigCheckResult(namedtuple('ConfigCheckResult', ['valid', 'missing', 'optional_set', 'optional_not_set', 'exp_type'])):
    __slots__ = ()
    # https://stackoverflow.com/questions/7914152/can-i-overwrite-the-string-form-of-a-namedtuple#comment32362249_7914212

    def __str__(self):
        valid, missing, optional_set, optional_not_set, exp_type = self
        return "experiment: '%s'; \nvalid: %s; \nmissing: %s; optional (set): %s; optional (not set): %s;" % (exp_type, valid, ', '.join(missing), ', '.join(optional_set), ', '.join(optional_not_set))

    def __repr__(self):
        s = str(self)
        return s.replace('; ', '\n\t - ')
