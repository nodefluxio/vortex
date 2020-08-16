import os
import sys
import warnings
from pathlib import Path, PurePath
from typing import Union
import shutil
import torchvision.datasets
from pprint import PrettyPrinter
from .torchvision import create_torchvision_dataset,SUPPORTED_TORCHVISION_DATASETS

_file_path = Path(__file__)
_default_dataset_path = os.path.join(
    os.getcwd(), "external", "datasets")
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

pp = PrettyPrinter(indent=4)

all_datasets = {
    'torchvision.datasets': SUPPORTED_TORCHVISION_DATASETS,
    'external': []
}
supported_dataset = {
    torchvision.datasets: SUPPORTED_TORCHVISION_DATASETS
}


def register_dvc_dataset(module: str, path: Union[str, Path] = _default_dataset_path):
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
            warnings.warn(
                "Found dataset %s contain '-' symbol, system automatically replace it with '_' symbol!" % (module))
            transformed_name = module.replace('-', '_')
            shutil.move(PurePath(_default_dataset_path, module),
                        PurePath(_default_dataset_path, transformed_name))
            module = transformed_name
        exec('from %s.utils import dataset as %s' % (module, module))
    except Exception as e:
        warnings.warn(
            'failed to import dataset %s, original error message is "%s"' % (module, str(e)))
        return
    py_module = eval('%s' % module)
    module_attributes = py_module.__dict__.keys()
    for attribute in _required_dataset_module_attributes:
        if not attribute in module_attributes:
            warnings.warn('skipping dataset %s' % module)
            return
    warnings.warn('adding %s to available datasets' % module)
    all_datasets['external'] += py_module.supported_dataset
    supported_dataset[py_module] = py_module.supported_dataset


def _scan_dvc_dataset(path: Path, dataset_env: str = 'VORTEX_DATASET_ROOT'):
    """
    given path, find all standardized dvc dataset
    """
    path = os.environ.get(dataset_env, path)
    ## python >= 3.6
    # path : Path = Path(path)
    path = Path(path)
    if path.exists() and path.is_dir():
    # if not path.exists():
    #     raise RuntimeError(
    #         'trying to scan dataset from %s, but the path doesnt exists' % str(path))
    # if not path.is_dir():
    #     raise RuntimeError(
    #         'trying to scan dataset from %s, but the path is not a directory' % str(path))
        warnings.warn('scanning for dataset at: %s' % str(path))
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
        warnings.warn("'external/datasets/' directory is not found!, skipping external dataset scanning!")

    warnings.warn('finished scanning dataset, available dataset(s) : \n%s' %
                  (pp.pformat(all_datasets)))


def get_base_dataset(dataset: str, dataset_args: dict = {}):
    flatten_datasets = [
        dataset for sublist in all_datasets.values() for dataset in sublist]
    if not dataset in flatten_datasets:
        raise RuntimeError("dataset %s not available, available dataset \n%s" % (
            dataset, pp.pformat(all_datasets)))
    for py_module, datasets in supported_dataset.items():
        if dataset in datasets:
            if py_module is torchvision.datasets:
                dataset = create_torchvision_dataset(dataset, dataset_args)
                return dataset
            else:
                return py_module.create_dataset(**dataset_args)
    raise RuntimeError("unexpected error")


# automatically scans for available datasets in default dataset
_scan_dvc_dataset(_default_dataset_path)
