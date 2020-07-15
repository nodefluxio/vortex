from vortex.core.factory import create_dataset
from vortex.utils.data.dataset.wrapper import BasicDatasetWrapper, DefaultDatasetWrapper
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
from vortex.utils.data.collater import create_collater

from torch.utils.data.dataloader import DataLoader


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
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.BGR)
        self.pad = ops.Paste(device='gpu', fill_value=0,
                             ratio=1, min_canvas_size=480, paste_x=0, paste_y=0)
        self.fixed_resize = ops.Resize(
            device='gpu', interp_type=types.DALIInterpType.INTERP_CUBIC, resize_longer=480)
        self.peek_shape = ops.Shapes(device='gpu')
        self.batch_pad = ops.Pad(device='cpu',axes=(0,1),fill_value=-1)

    def define_graph(self):
        data = self.source()

        graph_data=dict(zip(self.data_layout, data))
        images=self.decode(graph_data['images'])
        images = self.fixed_resize(images)
        pre_padded_image_shapes = self.peek_shape(images)
        images = self.pad(images)

        returned_data=dict()

        for layout in self.data_layout:
            if layout!='images':
                returned_data[layout]=self.batch_pad(graph_data[layout])
        # labels = tuple(graph_data[layout] for layout in self.data_layout if layout!='images')
        returned_data['images']=images
        returned_data = tuple(returned_data[layout] for layout in self.data_layout)
        return returned_data+(pre_padded_image_shapes,)

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
    def __iter__(self):
        return self
        
    def __next__(self):
        output = self.dali_pytorch_loader.__next__()[0] # Vortex doesn't support multiple pipelines yet

        # Prepare Pytorch style data loader output
        batch = []
        for i in range(len(output['images'])):
            image = output['images'][i]

            # DALI still have flaws about padding image to square, this is the workaround by bringing the image shape before padding
            pre_padded_image_size = output['pre_padded_image_shape'][i].cpu()[:2].type(torch.float32)
            padded_image_size = torch.tensor(image.shape[:2]).type(torch.float32)
            diff_ratio = pre_padded_image_size/padded_image_size

            # Prepare labels array
            labels_dict = dict()
            for layout in self.output_layout:
                label_output=output[layout][i].numpy()

                # Remove padded value from DALI, this assume that labels dimension 1 shape is same
                rows_with_padded_value=np.unique(np.where(label_output==-1)[0])
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

                    # Put back augmented labels in the placeholder array for returned labels
                    np.put_along_axis(ret_targets, values=augmented_label, axis=label_data_format['axis'],
                            indices=np.array(label_data_format['indices'])[np.newaxis, :])

            if list(self.data_format.keys())==['class_label']:
                ret_targets = ret_targets.flatten()
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
        diff_ratio=diff_ratio.numpy()
        if label_key == 'bounding_box':
            labels[:,::2] = labels[:,::2]*diff_ratio[1]
            labels[:,1::2] = labels[:,1::2]*diff_ratio[0]
        elif label_key == 'landmarks':
            pass
        return labels

if __name__ == "__main__":
    preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [0,0,0],
            'std' : [1, 1, 1]
        },
        'scaler' : 1
    })

    # dataset_config = EasyDict(
    #     {
    #         'train': {
    #             'dataset' : 'VOC2007DetectionDataset',
    #             'args' : {
    #                 'image_set' : 'train'
    #             }
    #         }
    #     }
    # )

    dataset_config = EasyDict(
        {
            'train': {
                'dataset': 'ImageFolder',
                'args': {
                    'root': 'tests/test_dataset/train'
                },
            }
        }
    )


    batch_size=64
    dataset = create_dataset(dataset_config, stage="train", preprocess_config=preprocess_args,wrapper_format='basic')
    collater_args = {'dataformat' : dataset.data_format}
    # collate_fn = create_collater('SSDCollate',**collater_args)
    collate_fn = None
    dataloader = DALIDataloader(dataset,
                 batch_size = 2,
                 num_thread = 1,
                 device_id = 0,
                 collate_fn=collate_fn,
                 shuffle=False)

    for datas in dataloader:
        dali_data = datas
        break
    
    # test vis
    # import cv2
    
    # vis = dali_data[0][0].cpu().numpy().copy()
    # h,w,c = vis.shape

    # allbboxes = dali_data[1][0][:,:4]

    # for bbox in allbboxes:
    #     x = int(bbox[0]*w)
    #     y = int(bbox[1]*h)
    #     x2 = int(bbox[2]*w)
    #     y2 = int(bbox[3]*h)
    #     cv2.rectangle(vis, (x, y),(x2, y2), (0, 0, 255), 2)

    dataset = create_dataset(dataset_config, stage="train", preprocess_config=preprocess_args,wrapper_format='default')

    dataloader_module_args = {
        'num_workers' : 0,
        'batch_size' : 4,
        'shuffle' : False,
    }

    dataloader = DataLoader(dataset, collate_fn=collate_fn, **dataloader_module_args)

    for datas in dataloader:
        pytorch_data = datas
        break
    import pdb; pdb.set_trace()
    # py_vis = pytorch_data[0][0].cpu().numpy().copy()
    # py_vis = np.transpose(py_vis, (1,2,0)).copy()

    # h,w,c = py_vis.shape

    # allbboxes = pytorch_data[1][0][:,:4]
    # for bbox in allbboxes:
    #     x = int(bbox[0]*w)
    #     y = int(bbox[1]*h)
    #     x2 = int(bbox[2]*w)
    #     y2 = int(bbox[3]*h)

    #     cv2.rectangle(py_vis, (x, y),(x2, y2), (0, 0, 255), 2)
    # cv2.imshow('dali', vis)
    # cv2.imshow('pytorch', py_vis)
    # cv2.waitKey(0)

