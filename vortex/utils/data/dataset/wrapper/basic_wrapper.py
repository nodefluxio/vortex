from typing import Dict, List, Union, Callable, Tuple
from easydict import EasyDict
import numpy as np
import cv2
from pathlib import Path

from ..dataset import get_base_dataset

KNOWN_DATA_FORMAT = ['class_label', 'bounding_box', 'landmarks']

class BasicDatasetWrapper():

    def __init__(self,
                 dataset: str,
                 stage: str,
                 preprocess_args: Union[EasyDict, dict],
                 augmentations: Union[Tuple[str, dict]],
                 dataset_args: Union[EasyDict, dict] = {},
                 ):
        """A basic form of dataset wrapper, maintain type checking and data format from integrated dataset

        Args:
            dataset (str): dataset name
            stage (str): pipeline stage, 'train' or 'validate' to differentiate augmentation process
            preprocess_args (Union[EasyDict, dict]): data preprocessinga arguments
            augmentations (Union[Tuple[str, dict]]): augmentations configuration
            dataset_args (Union[EasyDict, dict], optional): dataset initialization arguments. Defaults to {}.
        """
        self.stage = stage
        self.preprocess_args = preprocess_args
        self.augmentations_list = augmentations
        self.image_auto_pad = True

        if stage == 'train' and self.augmentations_list is not None:
            if not isinstance(self.augmentations_list, List):
                raise TypeError('expect augmentations config type as a list, got %s' % type(self.augmentations_list))

            # Handle multiple declaration of same augmentation modules
            aug_module_sequence = [augment.module for augment in self.augmentations_list]
            duplicate_modules = []
            for module in list(set(aug_module_sequence)):
                module_count = aug_module_sequence.count(module)
                if module_count > 1:
                    duplicate_modules.append(module)

            if len(duplicate_modules)!=0:
                raise RuntimeError('Detected duplicate declaration of augmentation modules in experiment file "augmentations" section, duplicate modules found = {}'.format(duplicate_module))

        self.dataset = get_base_dataset(
            dataset, dataset_args=dataset_args)
        if not 'data_format' in self.dataset.__dict__.keys():
            raise RuntimeError("expects dataset `%s` to have `data_format` field : dict<str,dict>, explaining data format (e.g. bounding_box, class_label, landmarks etc)" % dataset)
        
        ## make class_names optional

        # if not 'class_names' in self.dataset.__dict__.keys():
        #     raise RuntimeError("expects dataset `%s` to have `class_names` field : list<str>, explaining class string names which also map to its index positioning in the list. E.g. self.class_names = ['cat','dog'] means class_label = 0 is 'cat' " % dataset)

        self.class_names = None if not hasattr(self.dataset, 'class_names') else self.dataset.class_names

        self.data_format = EasyDict(self.dataset.data_format)
        # Data format standard check
        self.data_format = check_data_format_standard(self.data_format)
        self.preprocess_args = preprocess_args
        if self.stage == 'train':
            assert hasattr(self.preprocess_args, "input_size"), "at stage 'train', 'input_size' is mandatory for preprocess, please specify 'input_size' at 'preprocess_args'"
            assert hasattr(self.preprocess_args, "input_normalization"), "at stage 'train', 'input_normalization' is mandatory for preprocess, please specify 'input_normalization' at 'preprocess_args'"
            assert all([key in self.preprocess_args.input_normalization.keys() for key in ['std', 'mean']]), "at stage 'train', please specify 'mean' and 'std' for 'input_normalization'"
            assert isinstance(self.preprocess_args.input_size, int)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        if isinstance(image, str):
            if not Path(image).is_file():
                raise RuntimeError("Image file at '%s' not found!! Please check!" % (image))
        
        # Support if provided target only an int, convert to np.array
        # Classification case doesn't need any array slicing so constructed np array only array of 1 is enough
        if isinstance(target, int):
            target = np.array([target])

            if self.dataset.data_format['class_label'] is not None:
                warnings.warn("'int' type target should be paired with 'class_label' data_format with value None. Updating config..")
                self.dataset.data_format['class_label']=None
        
        # Target must be numpy array
        if not isinstance(target, np.ndarray):
            raise RuntimeError("Unknown target return of %s" % type(target))

        return image,target
    
    def disable_image_auto_pad(self):
        self.image_auto_pad = False

def check_data_format_standard(data_format: EasyDict):
    def _convert_indices_to_list(indices: EasyDict):
        '''
        Convert 'indices' in dictionary dict with 'start' and 'end' keys to list of int
        '''
        converted_indices = [*range(indices.start, indices.end+1)]
        return converted_indices
    if data_format:
        # Check if data_format keys is acceptable
        if not all([key in KNOWN_DATA_FORMAT for key in data_format]):
            raise RuntimeError("Unknown data format found! Please check! Known data format = %s, found %s" % (
                KNOWN_DATA_FORMAT, data_format.keys()))
        for key in data_format:
            # Check if data_format key value is None other than 'class_label'
            if data_format[key] is None and key != 'class_label':
                raise RuntimeError("Only 'class_label' data format can contain None value, found '%s' contain None" % key)
            if data_format[key] is not None:
                # Check if 'indices' exist in each data_format
                if 'indices' not in data_format[key]:
                    raise RuntimeError("'indices' key not found in '%s', please check!, found %s" % (key, data_format[key].keys()))
                # Check if data_format 'indices' type must be a list or a dict
                if type(data_format[key].indices) not in [list, EasyDict]:
                    raise RuntimeError("Expected '%s' 'indices' value type as list or dict, found %s" % (key, type(
                        data_format[key].indices)))
                # If 'indices' is dict, check for 'start' and 'end' keys
                if isinstance(data_format[key].indices, EasyDict):
                    # Convert 'indices' to list of int
                    if 'start' in data_format[key].indices and 'end' in data_format[key].indices:
                        data_format[key].indices = _convert_indices_to_list(
                            data_format[key].indices)
                    else:
                        raise RuntimeError("Expected '%s' 'indices' in dictionary type contain 'start' and 'end' keys, found %s please check!!" % (key, data_format[key].indices.keys()))
            # Bounding box data format standard checking
            if key == 'bounding_box':
                # Assume one class object detection if 'class_label' not found
                if 'class_label' not in data_format:
                    data_format.class_label = None
                # Check 'indices' length, must exactly 4 represent x,y,w,h
                if len(data_format.bounding_box.indices) != 4:
                    raise RuntimeError("'bounding_box' 'indices' data format must have length of 4!, found %s length of %i" % (
                        data_format.bounding_box.indices, len(data_format.bounding_box.indices)))

            # Landmarks data format standard checking
            if key == 'landmarks':
                # Check 'indices' length, must be in even number
                if len(data_format.landmarks.indices) % 2 != 0:
                    raise RuntimeError("'landmarks' 'indices' length must be an even number to be sliced equally into x,y coords! Found length of the 'indices' equal to %i ! Please check !" % (
                        len(data_format.landmarks.indices)))
                # Check 'asymm_pairs' format
                if 'asymm_pairs' in data_format.landmarks:
                    # Check type
                    if not isinstance(data_format.landmarks.asymm_pairs, list):
                        raise RuntimeError("'landmarks' 'asymm_pairs' value type must be a list!, found %s" % type(
                            data_format.landmarks.asymm_pairs))
                    # Check if 'asymm_pairs' contain empty list
                    if len(data_format.landmarks.asymm_pairs) == 0:
                        raise RuntimeError("'landmarks' 'asymm_pairs' contain 0 pairs! Please check!")
                    # Check pairs type inside the list
                    for pair in data_format.landmarks.asymm_pairs:
                        if not isinstance(pair, list):
                            raise RuntimeError("'landmarks' 'asymm_pairs' pairs value type inside the list must be a list!, found %s" % type(pair))
                        if len(pair) != 2 or not all([isinstance(value, int) for value in pair]):
                            raise RuntimeError("'landmarks' 'asymm_pairs' pairs value type inside the list must be list of int with the length of 2!, found %s" % pair)

        return data_format
    else:
        raise RuntimeError('Empty data_format dictionary! Please check!')


