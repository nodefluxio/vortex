from ..dataset.wrapper import BasicDatasetWrapper, DefaultDatasetWrapper
from easydict import EasyDict
from typing import Type, Union
from pathlib import Path
import random
import numpy as np
from collections import OrderedDict
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import copy
import torch
from ..augment import create_transform

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
        self.image_decode = ops.ImageDecoder(device = "mixed", output_type = types.BGR)
        self.label_cast = ops.Cast(device="cpu",
                             dtype=types.FLOAT)
        self.labels_pad_value = -99

        # Standard Augments Resize and Pad to Square
        preprocess_args = dataset_iterator.dataset.preprocess_args
        if 'mean' not in preprocess_args.input_normalization:
            preprocess_args.input_normalization.mean=[0.,0.,0.]
        if 'std' not in preprocess_args.input_normalization:
            preprocess_args.input_normalization.mean=[1.,1.,1.]
        if 'scaler' not in preprocess_args.input_normalization:
            preprocess_args.input_normalization.scaler=255

        standard_augment = EasyDict()
        standard_augment.transforms = [
            {'transform': 'StandardAugment','args':{'scaler' : preprocess_args.input_normalization.scaler,
                                                    'mean' : preprocess_args.input_normalization.mean,
                                                    'std' : preprocess_args.input_normalization.std,
                                                    'input_size' : preprocess_args.input_size,
                                                    'image_pad_value' : 0,
                                                    'labels_pad_value' : self.labels_pad_value
                                                    }
            }
        ]
        self.standard_augments = create_transform(
            'nvidia_dali', **standard_augment)
        # self.standard_augment = DALIStandardAugment(scaler = preprocess_args.input_normalization.scaler,
        #                                             mean = preprocess_args.input_normalization.mean,
        #                                             std = preprocess_args.input_normalization.std,
        #                                             input_size = preprocess_args.input_size,
        #                                             image_pad_value = 0,
        #                                             labels_pad_value = self.labels_pad_value)

    def define_graph(self):
        pipeline_input = self.source()

        # Prepare pipeline input data
        data = EasyDict()
        graph_data=dict(zip(self.data_layout, pipeline_input))
        data.images = self.image_decode(graph_data['images'])
        data.labels = EasyDict()
        for layout in self.data_layout:
            if layout!='images':
                data.labels[layout]=self.label_cast(graph_data[layout])
        data.additional_info=EasyDict()
        data.data_layout=self.data_layout
        # Custom Augmentation Start

        # Custom Augmentation End
        data = self.standard_augments(**data)

        # Prepare pipeline output tuple
        pipeline_output = []
        
        # Iterate new layout
        new_layout = data.data_layout

        for component in new_layout:
            if component == 'images':
                pipeline_output.append(data[component])
            elif component in data.labels:
                pipeline_output.append(data.labels[component])
            elif component in data.additional_info:
                pipeline_output.append(data.additional_info[component])

        pipeline_output = tuple(pipeline_output)
        return pipeline_output

class DALIDataloader():
    def __init__(self,
                 dataset,
                 batch_size,
                 num_thread = 1,
                 device_id = 0,
                 collate_fn = None,
                 shuffle = True
                 ):
        iterator = DALIIteratorWrapper(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       device_id=device_id)

        pipeline = DALIExternalSourcePipeline(dataset_iterator = iterator,
                                              batch_size=batch_size,
                                              num_threads=num_thread,
                                              device_id=device_id)
        self.dataset = iterator.dataset
        self.labels_pad_value = pipeline.labels_pad_value
        self.data_format = dataset.data_format
        self.output_layout = copy.copy(iterator.data_layout)
        self.output_layout.remove('images')
        # Additional field to retrieve image shape
        self.output_map = iterator.data_layout+['pre_padded_image_shape']
        self.dali_pytorch_loader = DALIGenericIterator(pipelines=[pipeline],
                                                        output_map=self.output_map,
                                                        size=iterator.size,
                                                        dynamic_shape=True,
                                                        fill_last_batch=False,
                                                        last_batch_padded=True,
                                                        auto_reset=True)
        self.collate_fn = collate_fn
        self.size = self.dali_pytorch_loader.size
        self.batch_size = batch_size
        self.preprocess_args = iterator.dataset.preprocess_args
    def __iter__(self):
        return self
        
    def __next__(self):
        output = self.dali_pytorch_loader.__next__()[0] # Vortex doesn't support multiple pipelines yet

        # Prepare Pytorch style data loader output
        batch = []
        for i in range(len(output['images'])):
            image = output['images'][i].type(torch.float32)

            # DALI still have flaws about padding image to square, this is the workaround by bringing the image shape before padding
            pre_padded_image_size = output['pre_padded_image_shape'][i].cpu()[:2].type(torch.float32)
            input_size = self.preprocess_args.input_size
            padded_image_size = torch.tensor([input_size,input_size]).type(torch.float32)
            diff_ratio = pre_padded_image_size/padded_image_size

            # Prepare labels array
            labels_dict = dict()
            for layout in self.output_layout:
                label_output=output[layout][i].numpy()

                # Remove padded value from DALI, this assume that labels dimension 1 shape is same
                rows_with_padded_value=np.unique(np.where(label_output==self.labels_pad_value)[0])
                label_output=np.delete(label_output,rows_with_padded_value,axis=0)
                
                # Placeholder to combine all labels
                if layout == 'original_labels':
                    ret_targets = label_output
                else:
                    # DALI still have flaws about padding image to square, this is the workaround by bringing the image shape before padding
                    label_output = self._fix_coordinates(label_output,layout,diff_ratio)
                    labels_dict[layout] = label_output
            
            # Modify labels placeholder with augmented labels
            for label_key in self.data_format:
                if label_key in self.output_layout:
                    label_data_format=self.data_format[label_key]
                    augmented_label = labels_dict[label_key]

                    # Refactor reshaped landmarks
                    if label_key == 'landmarks':
                        nrof_obj_landmarks = int(augmented_label.size / len(self.data_format['landmarks']['indices']))
                        augmented_label = augmented_label.reshape(nrof_obj_landmarks,len(self.data_format['landmarks']['indices']))

                    # Put back augmented labels in the placeholder array for returned labels
                    np.put_along_axis(ret_targets, values=augmented_label, axis=label_data_format['axis'],
                            indices=np.array(label_data_format['indices'])[np.newaxis, :])

            if list(self.data_format.keys())==['class_label']:
                ret_targets = ret_targets.flatten().astype('int')
            batch.append((image,torch.tensor(ret_targets)))

        if self.collate_fn is None:
            self.collate_fn = torch.utils.data._utils.collate.default_collate
        return self.collate_fn(batch)

    def __len__(self):
        if self.size%self.batch_size==0:
            return self.size//self.batch_size
        else:
            return self.size//self.batch_size+1

    def _fix_coordinates(self,labels,label_key,diff_ratio):
        """Fix coordinates label after image padding which break original image wh ratio

        Args:
            labels ([type]): [description]
            label_key ([type]): [description]
            diff_ratio ([type]): [description]

        Returns:
            [type]: [description]
        """

        diff_ratio=diff_ratio.numpy()
        labels[:,::2] = labels[:,::2]*diff_ratio[1]
        labels[:,1::2] = labels[:,1::2]*diff_ratio[0]
        return labels

supported_loaders = [('DALIDataLoader','basic')]

def create_loader(*args,**kwargs):
    return DALIDataloader(*args,**kwargs)