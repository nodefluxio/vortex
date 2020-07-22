import nvidia.dali.ops as ops
import nvidia.dali.types as types
from easydict import EasyDict
from typing import Union,Tuple,List
import numpy as np

__all__ = ['StandardAugment',
           'HorizontalFlip',
           'VerticalFlip',
           'RandomBrightnessContrast',
           'RandomJitter',
           'RandomHueSaturationValue']

class StandardAugment():
    def __init__(self,
                 input_size,
                 scaler = 255,
                 mean = [0.,0.,0.],
                 std = [1.,1.,1.],
                 image_pad_value = 0,
                 labels_pad_value = -99):

        # By default, CropMirrorNormalize divide each pixel by 255, to make it similar with Pytorch Loader behavior
        # in which we can control the scaler, we add additional scaler to reverse the effect
        self.image_normalize = ops.CropMirrorNormalize(
            device='gpu', mean=[value*255 for value in mean], std=[value*255 for value in std],
            output_layout='CHW',
            image_type=types.DALIImageType.BGR)

        self.scaler = ops.Normalize(
            device='gpu',
            scale = float(255/scaler),
            mean= 0,
            stddev = 1
        )
        
        # Padding and resize to prepare tensor output
        self.image_pad = ops.Paste(device='gpu', fill_value=image_pad_value,
                             ratio=1, min_canvas_size=input_size, paste_x=0, paste_y=0)
        self.labels_pad = ops.Pad(device='cpu',axes=(0,1),fill_value=labels_pad_value)
        
        self.model_input_resize = ops.Resize(
            device='gpu', interp_type=types.DALIInterpType.INTERP_CUBIC, resize_longer=input_size)
        self.peek_shape = ops.Shapes(device='gpu')

    def __call__(self,**data):
        data = EasyDict(data)
        images = data.images
        
        # Resize to model input size
        images = self.model_input_resize(images)
        
        # Save original shape before padding to fix label's with coordinates
        pre_padded_image_shapes = self.peek_shape(images)

        # Pad to square
        images = self.image_pad(images)
        
        # Normalize input
        images = self.image_normalize(images)
        images = self.scaler(images)
        
        # Labels padding
        for label_key in data.labels:
            data.labels[label_key] = self.labels_pad(data.labels[label_key])

        # Prepare result
        data.images=images

        # Add aditional information
        data.additional_info.pre_padded_image_shapes = pre_padded_image_shapes
        data.data_layout.append('pre_padded_image_shapes')
        return data

class HorizontalFlip():
    def __init__(self,p=0.5):
        self.flip_coin_hflip = ops.CoinFlip(probability=p)
        self.image_hflip = ops.Flip(device='gpu')
        self.bbox_hflip = ops.BbFlip(device='cpu')
        self.ldmrks_hflip = ops.CoordFlip(layout='xy',device='cpu')

    def __call__(self,**data):
        data = EasyDict(data)
        hflip_coin = self.flip_coin_hflip()
        
        # Flip data
        data.images = self.image_hflip(data.images,horizontal=hflip_coin)
        if 'bounding_box' in data.labels:
            data.labels.bounding_box = self.bbox_hflip(data.labels.bounding_box,horizontal=hflip_coin)
        if 'landmarks' in data.labels:
            data.labels.landmarks = self.ldmrks_hflip(data.labels.landmarks,flip_x=hflip_coin)

        # Store additional information
        
        # Get flip_flag from coin_flip to know which data is being flipped
        flip_flags = [int(key.split('flip_flag_')[1]) for key in data.additional_info.keys() if key.startswith('flip_flag_')]

        if len(flip_flags) == 0:
            flip_flag_keyname = 'flip_flag_0'
        else:
            flip_flags.sort()
            flip_flag_keyname = 'flip_flag_'+str(flip_flags[-1]+1)
        data.additional_info[flip_flag_keyname] = hflip_coin
        data.data_layout.append(flip_flag_keyname)
        
        return data

class VerticalFlip():

    def __init__(self,p=0.5):
        self.flip_coin_vflip = ops.CoinFlip(probability=p)
        self.image_vflip = ops.Flip(device='gpu',horizontal=0)
        self.bbox_vflip = ops.BbFlip(device='cpu',horizontal=0)
        self.ldmrks_vflip = ops.CoordFlip(layout='xy',device='cpu',flip_x=0)

    def __call__(self,**data):
        data = EasyDict(data)
        vflip_coin = self.flip_coin_vflip()
        
        # Flip data
        data.images = self.image_vflip(data.images,vertical=vflip_coin)
        if 'bounding_box' in data.labels:
            data.labels.bounding_box = self.bbox_vflip(data.labels.bounding_box,vertical=vflip_coin)
        if 'landmarks' in data.labels:
            data.labels.landmarks = self.ldmrks_vflip(data.labels.landmarks,flip_y=vflip_coin)

        # Store additional information
        
        # Get flip_flag from coin_flip to know which data is being flipped
        flip_flags = [int(key.split('flip_flag_')[1]) for key in data.additional_info.keys() if key.startswith('flip_flag_')]

        if len(flip_flags) == 0:
            flip_flag_keyname = 'flip_flag_0'
        else:
            flip_flags.sort()
            flip_flag_keyname = 'flip_flag_'+str(flip_flags[-1]+1)
        data.additional_info[flip_flag_keyname] = vflip_coin
        data.data_layout.append(flip_flag_keyname)

        return data

class RandomBrightnessContrast():
    
    def __init__(self,
                 p = .5,
                 brightness_limit : Union[List,float] = 0.5,
                 contrast_limit : Union[List,float] = 0.5):
        brightness_limit = _check_and_convert_limit_value(brightness_limit)
        contrast_limit = _check_and_convert_limit_value(contrast_limit)
        self.brightness_uniform = ops.Uniform(range=brightness_limit)
        self.contrast_uniform = ops.Uniform(range=contrast_limit)
        self.random_brightness_contrast = ops.BrightnessContrast(device='gpu')

        self.rng = ops.CoinFlip(probability = p)
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)

    def __call__(self,**data):
        data = EasyDict(data)

        brightness_rng = self.brightness_uniform()
        contrast_rng = self.contrast_uniform()

        aug_images = self.random_brightness_contrast(data.images,
                                                      brightness=brightness_rng,
                                                      contrast=contrast_rng)
        
        # DALI multiplexing to apply probability for applied augmentation or not
        apply_condition = self.bool(self.rng())
        neg_condition = apply_condition ^ True
        data.images = apply_condition * aug_images + neg_condition * data.images
        return data

    

class RandomJitter():

    def __init__(self,
                 p = .5,
                 nDegree = 2,
                 fill_value = 0.):

        self.jitter = ops.Jitter(device = 'gpu',
                                 fill_value = 0,
                                 nDegree = nDegree)
        
        self.flip_coin = ops.CoinFlip(probability=p)
    
    def __call__(self,**data):
        data = EasyDict(data)

        coin_flip = self.flip_coin()
        data.images=self.jitter(data.images,mask = coin_flip)

        return data

class RandomHueSaturationValue():
    
    def __init__(self,
                 p : .5,
                 hue_limit : Union[List,float] = 20.,
                 saturation_limit : Union[List,float] = .5,
                 value_limit : Union[List,float] = .5):
        
        hue_limit = _check_and_convert_limit_value(hue_limit,None,0)
        saturation_limit = _check_and_convert_limit_value(saturation_limit)
        value_limit = _check_and_convert_limit_value(value_limit)

        self.hsv = ops.Hsv(device='gpu')

        self.hue_uniform = ops.Uniform(range=hue_limit)
        self.saturation_uniform = ops.Uniform(range=saturation_limit)
        self.value_uniform = ops.Uniform(range=value_limit)

        self.rng = ops.CoinFlip(probability = p)
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)
    
    def __call__(self,**data):
        data = EasyDict(data)

        hue_rng = self.hue_uniform()
        saturation_rng = self.saturation_uniform()
        value_rng = self.value_uniform()

        aug_images = self.hsv(data.images,
                              hue = hue_rng,
                              saturation = saturation_rng,
                              value = value_rng)

        # DALI multiplexing to apply probability for applied augmentation or not
        apply_condition = self.bool(self.rng())
        neg_condition = apply_condition ^ True
        data.images = apply_condition * aug_images + neg_condition * data.images

        return data

def _check_and_convert_limit_value(value,minimum = 0,modifier = 1):
    if isinstance(value,List) or isinstance(value,Tuple):
        if len(value)!=2 or value[0]>value[-1]:
            raise ValueError('Limit must be provided as list/tuple with length 2 -> [min,max] value, \
                                found {}'.format(value))
    elif isinstance(value,int) or isinstance(value,float):
        value = [-value,value]
        value = np.array(value) + modifier
    else:
        import pdb; pdb.set_trace()    
        raise ValueError('Unknown limit type, expected to be list/tuple or int, found {}'.format(value))
    
    if minimum:
        if value[0] < minimum:
            raise ValueError('Minimum value limit is 0, found {}'.format(value[0]))

    value = value.tolist()
    return value