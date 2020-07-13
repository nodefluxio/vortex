from vortex.core.factory import create_dataset
from vortex.utils.data.dataset.wrapper import BasicDatasetWrapper, DefaultDatasetWrapper
from easydict import EasyDict
from typing import Type
from pathlib import Path
import random
import numpy as np
from collections import OrderedDict
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from timeit import default_timer as timer

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
        
        self.data_layout = ['images']+list(self.dataset.data_format.keys())

        # Get proper sharded data based on the device_id and total number of GPUs (world size) (For distributed training)
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
            sliced_labels = self.slice_labels(labels)
            for layout in self.data_layout:
                if layout == 'images':
                    with open(image, 'rb') as f:
                        return_data[layout].append(np.frombuffer(f.read(), dtype=np.uint8))
                else:
                    return_data[layout].append(sliced_labels[layout])
            self.i = (self.i + 1) % self.nrof_sharded_data
        return tuple(return_data[layout] for layout in self.data_layout)

    @property
    def size(self,):
        # return self.dataset_len
        return self.nrof_sharded_data

    def slice_labels(self,labels):
        sliced_labels = {}
        for label_name in self.dataset.data_format:
            sliced_labels[label_name]= np.take(labels, 
                                               axis=self.dataset.data_format[label_name].axis, 
                                               indices=self.dataset.data_format[label_name].indices)
        return sliced_labels

    next = __next__

class DALIExternalSourcePipeline(Pipeline):
    def __init__(self, dataset_iterator, batch_size, num_threads, device_id, seed = 12345):
        super().__init__(
                        batch_size,
                        num_threads,
                        device_id,
                        seed=seed)
        self.data_layout = dataset_iterator.data_layout
        self.source = ops.ExternalSource(source = dataset_iterator, 
                                         num_outputs = len(dataset_iterator.data_layout))
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.BGR)

    def define_graph(self):
        data = self.source()
        graph_data=dict(zip(self.data_layout, data))
        images=self.decode(graph_data['images'])
        labels = tuple(graph_data[layout] for layout in self.data_layout if layout!='images')

        return (images,)+labels

if __name__ == "__main__":
    preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [0.5, 0.5, 0.5],
            'std' : [0.5, 0.5, 0.5]
        },
        'scaler' : 255
    })

    dataset_config = EasyDict(
        {
            'train': {
            'dataset' : 'VOC2007DetectionDataset',
            'args' : {
            'image_set' : 'train'
            }
        }
        }
    )


    batch_size=128
    dataset = create_dataset(dataset_config, stage="train", preprocess_config=preprocess_args,wrapper_format='basic')
    iterator = DALIIteratorWrapper(dataset,batch_size=batch_size)
    pipeline = DALIExternalSourcePipeline(dataset_iterator = iterator,batch_size=batch_size,num_threads=4,device_id=0)
    pipeline.build()
    out=pipeline.run()
