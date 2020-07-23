from typing import Type
from ....dataset.wrapper import BasicDatasetWrapper, DefaultDatasetWrapper
from pathlib import Path
import random
from collections import OrderedDict
import numpy as np

class DALIIteratorWrapper(object):
    def __init__(self,
                 dataset : Type[BasicDatasetWrapper],
                 batch_size: int,
                 device_id: int = 0,
                 num_gpus: int = 1,
                 shuffle: bool = True):
        """An extended iterator to bridge dataset and NVIDIA DALI External Input Source Pipeline

        Arguments:
            dataset {object} -- A base dataset iterator object to return namedtuple data format
            batch_size {int} -- Batch size of data returned in every iteration

        Keyword Arguments:
            device_id {int} -- GPU ID to be use (default: {0})
            num_gpus {int} -- Total GPUs to be used (default: {1})
            shuffle {bool} -- Shuffle dataset (default: {True})

        Raises:
            TypeError: if base iterator dataset not returning namedtuple _fields
            ValueError: if 'image' field is not found on the returned value
            TypeError: if __len__ method not implemented in base iterator dataset
        """
        self.batch_size = batch_size
        self.dataset = dataset

        # Dataset check
        if not isinstance(self.dataset,BasicDatasetWrapper) or isinstance(self.dataset,DefaultDatasetWrapper):
            raise RuntimeError('DALI loader `dataset` args expect BasicDatasetWrapper object, found {}'.format(str(type(dataset))))
        
        if not isinstance(self.dataset[0][0],str) and not isinstance(self.dataset[0][0],Path):
            raise RuntimeError('DALI loader expect `dataset` image data as `str` or `Path` object of the image file, found {}'.format(type(self.dataset[0][0])))
        
        # Whole dataset size
        try:
            self.dataset_len = len(self.dataset)
        except:
            raise RuntimeError(
                "'dataset' __len__ is not implemented!!")
        
        # Pipeline data layout, only include labels with specified `data_format`
        # `original_labels` is unsliced labels
        self.data_layout = ['images']+[key for key in self.dataset.data_format.keys() if self.dataset.data_format[key] is not None]+['original_labels']

        # Get proper sharded data based on the device_id and total number of GPUs (world size) (For distributed training in future usage)
        self.sharded_idx = list(range(self.dataset_len * device_id //
                                      num_gpus, self.dataset_len * (device_id + 1) // num_gpus))
        if shuffle:
            random.shuffle(self.sharded_idx)
        self.nrof_sharded_data = len(self.sharded_idx)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        return_data = OrderedDict()
        for layout in self.data_layout:
            return_data[layout] = []

        for _ in range(self.batch_size):
            selected_index = self.sharded_idx[self.i]
            image,labels = self.dataset[selected_index]
            sliced_labels = self._slice_labels(labels)
            for layout in self.data_layout:
                if layout == 'images':
                    with open(image, 'rb') as f:
                        return_data[layout].append(np.frombuffer(f.read(), dtype=np.uint8))
                elif layout != 'original_labels':
                    return_data[layout].append(sliced_labels[layout])
                # Original labels array is also forwarded to add flexibility for labels with unsupported data_format
                # So it can still be forwarded to the pipeline output
                else:
                    if len(labels.shape)==1:
                        labels=labels[np.newaxis,:]
                    return_data[layout].append(labels)
            self.i = (self.i + 1) % self.nrof_sharded_data
        return tuple(return_data[layout] for layout in self.data_layout)

    @property
    def size(self,):
        return self.nrof_sharded_data

    def _slice_labels(self,labels):
        sliced_labels = {}

        for label_name in self.dataset.data_format:
            if self.dataset.data_format[label_name] is not None:
                sliced_labels[label_name]= np.take(labels, 
                                                axis=self.dataset.data_format[label_name].axis, 
                                                indices=self.dataset.data_format[label_name].indices)
                
                # Transform flattened landmarks coords into 2 dim array
                if label_name=='landmarks':
                    sliced_labels[label_name]=sliced_labels[label_name].reshape((int(sliced_labels[label_name].size/2), 2))
        
        ## Handle if labels contain only 'class_labels' and the data format is None
        return sliced_labels

    next = __next__