import os
import sys
import logging
import shutil
import torchvision.datasets

from pathlib import Path, PurePath
from typing import Union
from .torchvision import create_torchvision_dataset, SUPPORTED_TORCHVISION_DATASETS


logger = logging.getLogger(__name__)

_file_path = Path(__file__)
_file_dir  = _file_path.parent
_default_dataset_path = os.path.join(os.getcwd(), "external", "datasets")
_exclude_dirs = [
    '__pycache__'
]
_required_dirs = [
    'utils'
]
_required_dataset_module_attributes = [
    'supported_dataset',
    'create_dataset',
]

all_datasets = {
    'torchvision.datasets': SUPPORTED_TORCHVISION_DATASETS,
    'external': []
}
supported_dataset = {
    torchvision.datasets: SUPPORTED_TORCHVISION_DATASETS
}

def _format_dict(data: dict) -> str:
    formatted = ""
    for idx, (name, values) in enumerate(data.items()):
        if idx > 0:
            formatted += "\n"
        formatted += "{}:\n".format(name)
        for n, v in enumerate(values):
            formatted += "    {}".format(v)
            if n != len(values)-1 or idx != len(data)-1:
                formatted += "\n"
    return formatted


# TODO: rename function name, support passing module directly as python module
# NOTE: temporarily allow to skip submodule `utils`, TODO: cleanup by supporting python module directly
def register_dvc_dataset(module: str, path: Union[str, Path] = _default_dataset_path, submodule: Union[str,None]='utils'):
    """
    Register DVC type dataset to be recognized by Vortex.

    rules:
        - all dataset module should have `utils` directory
        - inside `utils`, `dataset.py` should be present
        - `dataset.py` should have `supported_dataset` which is list of str and `create_dataset` method
    """
    global supported_dataset, all_datasets

    if isinstance(path, Path):
        path = os.path.abspath(path)
    sys.path.append(path)
    try:
        # Added capability to automatic convert dir with '-' to '_'
        if len(module.split('-')) > 1:
            logger.info("Found dataset {} contain '-' symbol, automatically replace it "
                "with '_' symbol!".format(module))
            transformed_name = module.replace('-', '_')
            shutil.move(PurePath(_default_dataset_path, module),
                        PurePath(_default_dataset_path, transformed_name))
            module = transformed_name
        # TODO: find out exec alternative
        if submodule is None:
            # actual dataset doesn't put at submodule, import as it is
            exec('import %s as %s' % (module, module))
        else:
            # NOTE: assuming actual dataset is named after `dataset`
            exec('from %s.%s import dataset as %s' % (module, submodule, module))
    except Exception as e:
        logger.info('failed to import dataset {}, original error message is "{}"'.format(module, str(e)))
        return
    py_module = eval('%s' % module)
    module_attributes = py_module.__dict__.keys()
    for attribute in _required_dataset_module_attributes:
        if not attribute in module_attributes:
            logger.info('skipping dataset {}'.format(module))
            return
    logger.info('adding {} to available datasets'.format(module))
    all_datasets['external'] += py_module.supported_dataset
    supported_dataset[py_module] = py_module.supported_dataset


# TODO: rename function name,
def _scan_dvc_dataset(path: Path, dataset_env: str = 'VORTEX_DATASET_ROOT'):
    """
    given path, find all standardized dvc dataset
    """
    path = Path(os.environ.get(dataset_env, path))
    if path.exists() and path.is_dir():
        logger.info('scanning for dataset at: {}'.format(str(path)))
        for child in path.iterdir():
            if child.is_dir():
                dir_name = child.name
                if dir_name in _exclude_dirs:
                    continue
                child2 = [ch.name for ch in child.iterdir()]
                ok = all([req_dir in child2 for req_dir in _required_dirs])
                if ok:
                    register_dvc_dataset(dir_name, path=path)
    else:
        logger.info("{} directory is not found!, skipping external dataset "
            "scanning!".format(str(path)))

    logger.info('finished scanning dataset, available datasets: \n{}'
        .format(_format_dict(all_datasets)))


def get_base_dataset(dataset: str, dataset_args: dict = {}):
    flatten_datasets = [
        dataset for sublist in all_datasets.values() for dataset in sublist]
    if not dataset in flatten_datasets:
        raise RuntimeError("dataset {} not available, available datasets: \n{}"
            .format(dataset, _format_dict(all_datasets)))
    for py_module, datasets in supported_dataset.items():
        if dataset in datasets:
            if py_module is torchvision.datasets:
                dataset = create_torchvision_dataset(dataset, dataset_args)
                return dataset
            else:
                return py_module.create_dataset(**dataset_args)
    raise RuntimeError("unexpected error")


# TODO: support passing module directly as python module
register_dvc_dataset('darknet', _file_dir, submodule=None)
# automatically scans for available datasets in default dataset
_scan_dvc_dataset(_default_dataset_path)
