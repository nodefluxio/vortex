from easydict import EasyDict
from typing import Type, Union, List, Callable
from pathlib import Path
import random
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import copy
import torch
from ..dataset.wrapper.default_wrapper import check_and_fix_coordinates
from ....networks.modules.preprocess.normalizer import to_tensor,normalize
import cv2
from PIL import Image
import ray
from .modules.nvidia_dali import DALIIteratorWrapper,DALIExternalSourcePipeline
from ..augment import create_transform
from ..dataset.wrapper import BasicDatasetWrapper

class DALIDataloader():
    """DataLoader for Nvidia DALI augmentation pipeline,
    to handle non-DALI augmentations, this loader utilize Ray to paralellize augmentation
    for every sample in a batch
    """
    def __init__(self,
                 dataset : Type[BasicDatasetWrapper],
                 batch_size : int,
                 num_thread : int = 1,
                 device_id : int = 0,
                 collate_fn : Type[Callable] = None,
                 shuffle : bool = True
                 ):
        """Initialization

        Args:
            dataset (Type[BasicDatasetWrapper]): dataset object to be adapted into DALI format
            batch_size (int): How many samples per batch to load
            num_thread (int, optional): Number of CPU threads used by the pipeline. Defaults to 1.
            device_id (int, optional): GPU id to be used for pipeline. Defaults to 0.
            collate_fn (Type[Callable], optional): merges a list of samples to form a mini-batch of Tensor(s). Defaults to None.
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch. Defaults to True.
        """

        iterator = DALIIteratorWrapper(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       device_id=device_id)

        self.dataset = iterator.dataset
        self.data_format = dataset.data_format
        self.preprocess_args = iterator.dataset.preprocess_args

        # Initialize DALI only augmentations
        self.augmentations_list = self.dataset.augmentations_list

        dali_augments = None
        external_augments = None
        normalize = True
        if self.dataset.stage == 'train' and self.augmentations_list is not None:
            external_augments = []
            # Handler if using Nvidia DALI, if DALI augmentations is used in experiment file, it must be in the first order
            aug_module_sequence = [augment.module for augment in self.augmentations_list]
            if 'nvidia_dali' in aug_module_sequence and aug_module_sequence[0] != 'nvidia_dali':
                raise RuntimeError('Nvidia DALI augmentation module must be in the first order of the "augmentations" list!, found {}'.format(aug_module_sequence[0]))

            for augment in self.augmentations_list:
                module_name = augment.module
                module_args = augment.args
                if not isinstance(module_args, dict):
                    raise TypeError("expect augmentation module's args value to be dictionary, got %s" % type(module_args))
                tf_kwargs = module_args
                tf_kwargs['data_format'] = self.data_format
                augments = create_transform(module_name, **tf_kwargs)
                if module_name == 'nvidia_dali':
                    dali_augments = augments
                else:
                    external_augments.append(augments)

            self.external_executors = None
            # If there are any external augments 
            if len(external_augments)!=0:
                # do not apply normalization and channel format swap in DALI pipeline
                normalize = False

                # Instantiate external augments executor
                ray.init(ignore_reinit_error=True)
                transforms_list_ref = ray.put(external_augments)
                data_format_ref = ray.put(self.data_format)
                preprocess_args_ref = ray.put(self.preprocess_args)

                self.external_executors = [ExternalAugmentsExecutor.remote(transforms_list_ref,
                                                                  data_format_ref,
                                                                  preprocess_args_ref) for i in range(batch_size)]


        pipeline = DALIExternalSourcePipeline(dataset_iterator = iterator,
                                              batch_size=batch_size,
                                              num_threads=num_thread,
                                              device_id=device_id,
                                              dali_augments=dali_augments,
                                              normalize=normalize)
        self.labels_pad_value = pipeline.labels_pad_value
        self.original_data_layout = copy.copy(pipeline.original_data_layout)
        self.original_data_layout.remove('images')

        # Additional field to retrieve image shape
        self.output_map = pipeline.pipeline_output_data_layout
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
            aug_labels = dict()
            for layout in self.original_data_layout:
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
                    aug_labels[layout] = label_output

            # Modify labels placeholder with augmented labels
            for label_key in self.data_format:
                if label_key in self.original_data_layout:
                    label_data_format=self.data_format[label_key]
                    augmented_label = aug_labels[label_key]

                    # Refactor reshaped landmarks and apply asymmetric coordinates fixing if needed
                    if label_key == 'landmarks':
                        nrof_obj_landmarks = int(augmented_label.size / len(self.data_format['landmarks']['indices']))
                        
                        # Reshape to shape [nrof_objects,nrof_points] 
                        augmented_label = augmented_label.reshape(nrof_obj_landmarks,len(self.data_format['landmarks']['indices']))

                        # Coordinates sequence fixing for asymmetric landmarks
                        if 'asymm_pairs' in self.data_format['landmarks']:
                            # Extract flip flag from pipeline output
                            # import pdb; pdb.set_trace()
                            flip_flags = np.array([output[key][i].numpy() for key in output.keys() if key.startswith('flip_flag_')])
                            flip_count = np.sum(flip_flags)

                            # if flip count mod 2 is even, skip coordinates sequence flipping
                            if flip_count%2 == 1:
                                n_keypoints = int(len(augmented_label[0])/2)
                                augmented_label = augmented_label.reshape((-1, n_keypoints, 2))

                                # For each index keypoints pair, swap the position
                                for keypoint_pair in self.data_format.landmarks.asymm_pairs:
                                    keypoint_pair = np.array(keypoint_pair)
                                    augmented_label[:, keypoint_pair,
                                            :] = augmented_label[:, keypoint_pair[::-1], :]
                                    # Convert back to original format
                                augmented_label = augmented_label.reshape((-1, n_keypoints * 2))


                    # Put back augmented labels in the placeholder array for returned labels
                    np.put_along_axis(ret_targets, values=augmented_label, axis=label_data_format['axis'],
                            indices=np.array(label_data_format['indices'])[np.newaxis, :])

            if list(self.data_format.keys())==['class_label']:
                ret_targets = ret_targets.flatten().astype('int')
            batch.append((image,torch.tensor(ret_targets)))

        # Apply external (non-DALI) augments, utilizing ray
        if self.external_executors :
            batch = [(image.cpu(),target) for image,target in batch]
            batch_ref = ray.put(batch)
            batch_futures = [self.external_executors[index].run.remote(batch_ref,index) for index in range(len(batch))]
            batch = ray.get(batch_futures)
        #

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

@ray.remote
class ExternalAugmentsExecutor():
    """Ray actors to handle augmentation for every sample in a batch
    """
    def __init__(self,
                 transforms_list : list,
                 data_format : dict,
                 preprocess_args : dict):
        """Initialization

        Args:
            transforms_list (list): list of augmentation to be applied
            data_format (dict): dataset data format
            preprocess_args (dict): preprocess args for input normalization
        """

        self.augments = transforms_list
        self.data_format = data_format
        self.preprocess_args = preprocess_args

        # Standardized computer vision augmentation initialization, longest resize and pad to square
        standard_tf_kwargs = EasyDict()
        standard_tf_kwargs.data_format = self.data_format
        standard_tf_kwargs.transforms = [
            {'transform': 'LongestMaxSize', 'args': {'max_size': preprocess_args.input_size}},
            {'transform': 'PadIfNeeded', 'args': {'min_height': preprocess_args.input_size,
                                                  'min_width': preprocess_args.input_size,
                                                  'border_mode': cv2.BORDER_CONSTANT,
                                                  'value': [0, 0, 0]}},
        ]
        self.standard_augments = create_transform(
            'albumentations', **standard_tf_kwargs)

    def run(self,batch,index):
        data = batch[index]
        image = data[0].numpy()
        target = data[1].numpy()
        # Configured computer vision augment -- START
        if self.augments is not None:
            # Out of image shape coordinates adaptation -- START
            image, target = check_and_fix_coordinates(
                image, target, self.data_format)
            # Out of image shape coordinates adaptation --  END
            for augment in self.augments:
                image, target = augment(image, target)
                if target.shape[0] < 1:
                    raise RuntimeError("The configured augmentations resulting in 0 shape target!! Please check your augmentation and avoid this!!")
        if not isinstance(image, Image.Image) and not isinstance(image, np.ndarray):
            raise RuntimeError('Expected augmentation output in PIL.Image.Image or numpy.ndarray format, got %s ' % type(image))
        if (np.all(image >= 0.) and np.all(image <= 1.)) or isinstance(image, torch.Tensor):
            pixel_min, pixel_max = np.min(image), np.max(image)
            if pixel_min == 0. and pixel_max == 0.:
                raise RuntimeError('Augmentation image output producing blank image ( all pixel value == 0 ), please check and visualize your augmentation process!!')
            else:
                raise RuntimeError('Augmentation image output expect unnormalized pixel value (0-255), got min %2.2f and max %2.2f' % (pixel_min, pixel_max))
        # Configured computer vision augment -- END
        # Standard computer vision augment -- START
        image, target = self.standard_augments(image, target)

        input_normalization = self.preprocess_args.input_normalization
        if 'scaler' not in input_normalization:
            input_normalization.scaler=255
        image = torch.from_numpy(np.expand_dims(image, axis=0))
        image = to_tensor(image,scaler=input_normalization.scaler)
        image = normalize(
            image, input_normalization.mean, input_normalization.std).squeeze(0)
        # Standard computer vision augment -- END
        if not isinstance(target, torch.Tensor):
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
            else:
                raise RuntimeError(
                    'unsupported data type for target, got %s' % type(target))
        data = image, target
        return data

supported_loaders = [('DALIDataLoader','basic')] # (Supported data loader, Supported dataset wrapper format)

def create_loader(*args,**kwargs):
    return DALIDataloader(*args,**kwargs)