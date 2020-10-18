from pathlib import Path
from collections import namedtuple
from typing import Dict, List, Union, Callable, Tuple
from multipledispatch import dispatch

import cv2
import random
import numpy as np
import pandas as pd

_file_path = Path(__file__)
_repo_root = _file_path.parent.parent

__all__ = [
    'DarknetDataset',
]

supported_dataset = [
    'DarknetDataset'
]

class DarknetDataset :
    def __init__(self,
                 txt_path: str, img_root: str, names: str,
                 check_exists=True):
        """initialize DarknetDataset Object

        Args:
            txt_path (str): path to image list filename
            img_root (str): path to image root directory
            names (str): path to label filename
            check_exists (bool, optional): check the existence of image and labels. Defaults to True.

        Raises:
            RuntimeError: if some any image doesn't exist
            RuntimeError: if some any label doesn't exist
        """        
        self.image_filenames = []
        self.label_filenames = []
        missing_labels, missing_images = [], []
        txt_path = Path(txt_path).expanduser()
        img_root = Path(img_root).expanduser()
        names = Path(names).expanduser()
        img_paths = pd.read_fwf(txt_path, header=None)[0]
        for img_path in img_paths:
            img_filename = str(Path(txt_path).parent / img_root / img_path)
            self.image_filenames.append(img_filename)
            if not Path(img_filename).exists() :
                missing_images.append(img_filename)
            label_filename = img_filename.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            if not Path(label_filename).exists() :
                missing_labels.append(label_filename)
            self.label_filenames.append(label_filename)
        
        ## make sure all images and labels exists
        if len(missing_images) and check_exists:
            raise RuntimeError("some image(s) do not exists! %s" %'\n'.join(missing_images))
        if len(missing_labels) and check_exists:
            raise RuntimeError("some label(s) do not exists! %s" %'\n'.join(missing_labels))

        ## data format for single label
        self.data_format = {
            ## indices and axis for np.take
            'bounding_box' : {
                'indices' : [2,3,4,5],
                'axis' : 1,
            },
            ## indices and axis to take class label
            'class_label' : {
                'indices' : [1],
                'axis' : 1,
            }
        }

        self.class_names = []
        with names.open() as f:
            self.class_names = f.readlines()
        self.class_names = [name.rstrip() for name in self.class_names]

    def __len__(self):
        """return the lengt of datset

        Returns:
            int: number of dataset entry
        """        
        return len(self.image_filenames)

    def __getitem__(self, index):
        """random access to dataset

        Args:
            index (iint): index of dataset to be returned

        Returns:
            tuple: named tuple of image, target
        """        
        Data = namedtuple('Data', ['image', 'target'])
        image = self.image_filenames[index]
        image = cv2.imread(image)
        target = []

        with open(self.label_filenames[index]) as file:
            for line in file:
                class_label_and_bbox = line.strip().split(' ')
                class_label = int(class_label_and_bbox[0])
                x_cent, y_cent, w, h = np.asarray(class_label_and_bbox[1:], np.float32)
                x1 = x_cent - (w / 2)
                y1 = y_cent - (h / 2)
                # 0 acts as placeholder for index
                target.append([0, class_label, x1, y1, w, h])

        target = np.stack(target)
        return image, target


def create_dataset(*args, **kwargs) :
    """create dataset

    Returns:
        DarknetDataset: DarknetDataset object
    """    
    return DarknetDataset(*args, **kwargs)
