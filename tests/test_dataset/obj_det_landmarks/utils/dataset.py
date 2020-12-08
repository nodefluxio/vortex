import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Union

from .dvc import dvc_pull

_file_path = Path(__file__)
_repo_root = _file_path.parent.parent

__all__ = [
    'TestObjDetLandmarksDataset',
]

supported_dataset = [
    'TestObjDetLandmarksDataset'
]

class TestObjDetLandmarksDataset :
    ## available datasets
    dataset_version = 'v0.2'
    train_datasets = {
        'v0.2' : [
            str(_repo_root / 'train/data/labels.txt')
        ]
    }
    val_datasets = {
        'v0.2' : [
            str(_repo_root / 'validation/data/labels.txt')
        ]
    }
    train_img_roots = {
        'v0.2' : [
            'images/',
        ]
    }
    val_img_roots = {
        'v0.2' : [
            '../../train/data/images/',
        ]
    }
    """Face detection with retinaface format with dvc support

    Parameters:
        dvc_remote : which dvc remote to pull, if not None
        txt_paths : a list of labels
        img_roots : a list of image roots, relative to labels
    """
    def __init__(self, 
        dvc_remote : Union[str,None]='gcs', 
        train : bool=True) :

        version = TestObjDetLandmarksDataset.dataset_version
        txt_paths = TestObjDetLandmarksDataset.train_datasets[version] if train else TestObjDetLandmarksDataset.val_datasets[version]
        img_roots = TestObjDetLandmarksDataset.train_img_roots[version] if train else TestObjDetLandmarksDataset.val_img_roots[version]

        ## check existence
        txt_paths_exist = [
            Path(txt_path).exists()
            for txt_path in txt_paths
        ]
        if not all(txt_paths_exist) and dvc_remote :
            workdir = _repo_root
            print(workdir)
            result = dvc_pull(dvc_remote,workdir=str(workdir))
            print(result)
        ## TODO : work around for img roots if not exists (?)
        self.imgs_path = []
        self.words = []
        for txt_path, image_root in zip(txt_paths, img_roots) :
            f = open(txt_path,'r')
            filename = str(Path(txt_path).name)
            lines = f.readlines()
            isFirst = True
            labels = []
            for line in lines:
                line = line.rstrip()
                if line.startswith('#'):
                    if isFirst is True:
                        isFirst = False
                    else:
                        labels_copy = labels.copy()
                        self.words.append(labels_copy)
                        labels.clear()
                    path = line[2:]
                    path = txt_path.replace(filename,image_root+'/') + path
                    self.imgs_path.append(path)
                    if not Path(path).exists() :
                        raise RuntimeError("file %s not exists"%path)
                else:
                    line = line.split(' ')
                    label = [float(x) for x in line]
                    labels.append(label)

            self.words.append(labels)
        logging.info('dataset version : %s' %version)
        ## data format for single label
        self.data_format = {
            ## a indices and axis for np.take
            'bounding_box' : {
                'indices' : [0,1,2,3],
                'axis' : 1,
            },
            ## indices to take class label
            'class_label' : None,
            'landmarks' : {
                'indices' : [4,5,6,7,8,9,10,11,12,13],
                'axis' : 1,
                'asymm_pairs' : [[0,1],[3,4]],
            }
        }
        self.class_names = ['frontal_face']

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape
        img = self.imgs_path[index]

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[2]  # w
            annotation[0, 3] = label[3]  # h

            # landmarks
            try :
                annotation[0, 4] = label[4]    # l0_x
                annotation[0, 5] = label[5]    # l0_y
                annotation[0, 6] = label[7]    # l1_x
                annotation[0, 7] = label[8]    # l1_y
                annotation[0, 8] = label[10]   # l2_x
                annotation[0, 9] = label[11]   # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
            except :
                annotation[0, 4] = -1

            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            
            try :
                annotation[0, 0:14:2] /= width
                annotation[0, 1:14:2] /= height
            except :
                pass

            ## enforce x1, y1, w, h is valid
            annotation[0, 2] = max(max(0, annotation[0, 2]), 1e-3) # w
            annotation[0, 3] = max(max(0, annotation[0, 3]), 1e-3) # h
            annotation[0, 0] = max(0, annotation[0, 0])
            annotation[0, 1] = max(0, annotation[0, 1])
            x2, y2 = annotation[0, 0] + annotation[0, 2], annotation[0, 1] + annotation[0, 3]
            x2, y2 = min(1.0, x2), min(1.0, y2)
            annotation[0,2] = x2 - annotation[0,0]
            annotation[0,3] = y2 - annotation[0,1]
            annotation[0, 2] = max(0, annotation[0, 2]) # w
            annotation[0, 3] = max(0, annotation[0, 3]) # h
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        return img, target


class DefaultCollate :
    def __init__(self) :
        pass
    def __call__(self, batch) :
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
        """
        try :
            import torch
        except :
            raise RuntimeError("current implementation needs torch")
        targets = []
        imgs = []
        for _, sample in enumerate(batch):
            for _, tup in enumerate(sample):
                if torch.is_tensor(tup):
                    imgs.append(tup)
                elif isinstance(tup, type(np.empty(0))):
                    annos = torch.from_numpy(tup).float()
                    targets.append(annos)

        return (torch.stack(imgs, 0), targets)

def create_dataset(*args, **kwargs) :
    return TestObjDetLandmarksDataset(*args, **kwargs)
