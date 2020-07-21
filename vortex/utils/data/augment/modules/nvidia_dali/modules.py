import nvidia.dali.ops as ops
import nvidia.dali.types as types
from easydict import EasyDict

__all__ = ['StandardAugment',
           'HorizontalFlip',
           'VerticalFlip']

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
        self.flip_coin = ops.CoinFlip(probability=p)
        self.image_hflip = ops.Flip(device='gpu')
        self.bbox_hflip = ops.BbFlip(device='cpu')
        self.ldmrks_hflip = ops.CoordFlip(layout='xy',device='cpu')

    def __call__(self,**data):
        data = EasyDict(data)
        flip = self.flip_coin()
        
        # Flip data
        data.images = self.image_hflip(data.images,horizontal=flip)
        if 'bounding_box' in data.labels:
            data.labels.bounding_box = self.bbox_hflip(data.labels.bounding_box,horizontal=flip)
        if 'landmarks' in data.labels:
            data.labels.landmarks = self.ldmrks_hflip(data.labels.landmarks,flip_x=flip)

        # Store additional information
        
        # Get flip_flag from coin_flip to know which data is being flipped
        flip_flags = [int(key.split('flip_flag_')[1]) for key in data.additional_info.keys() if key.startswith('flip_flag_')]

        if len(flip_flags) == 0:
            flip_flag_keyname = 'flip_flag_0'
        else:
            flip_flags.sort()
            flip_flag_keyname = 'flip_flag_'+str(flip_flags[-1]+1)
        data.additional_info[flip_flag_keyname] = flip
        data.data_layout.append(flip_flag_keyname)
        
        return data

class VerticalFlip():
    def __init__(self,p=0.5):
        self.flip_coin = ops.CoinFlip(probability=p)
        self.image_hflip = ops.Flip(device='gpu')
        self.bbox_hflip = ops.BbFlip(device='cpu')
        self.ldmrks_hflip = ops.CoordFlip(layout='xy',device='cpu')

    def __call__(self,**data):
        data = EasyDict(data)
        flip = self.flip_coin()
        
        # Flip data
        data.images = self.image_hflip(data.images,vertical=flip)
        if 'bounding_box' in data.labels:
            data.labels.bounding_box = self.bbox_hflip(data.labels.bounding_box,vertical=flip)
        if 'landmarks' in data.labels:
            data.labels.landmarks = self.ldmrks_hflip(data.labels.landmarks,flip_y=flip)

        # Store additional information
        
        # Get flip_flag from coin_flip to know which data is being flipped
        flip_flags = [int(key.split('flip_flag_')[1]) for key in data.additional_info.keys() if key.startswith('flip_flag_')]

        if len(flip_flags) == 0:
            flip_flag_keyname = 'flip_flag_0'
        else:
            flip_flags.sort()
            flip_flag_keyname = 'flip_flag_'+str(flip_flags[-1]+1)
        data.additional_info[flip_flag_keyname] = flip
        data.data_layout.append(flip_flag_keyname)
        return data