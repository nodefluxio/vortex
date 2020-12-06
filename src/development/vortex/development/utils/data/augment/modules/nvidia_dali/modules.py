import nvidia.dali.ops as ops
import nvidia.dali.types as types

from easydict import EasyDict
from typing import Union, List
from .utils import _check_and_convert_limit_value

__all__ = ['StandardAugment',
           'HorizontalFlip',
           'VerticalFlip',
           'RandomBrightnessContrast',
           'RandomJitter',
           'RandomHueSaturationValue',
           'RandomWater',
           'RandomRotate']

class StandardAugment():
    """Standard augmentation which resize the input image into desired size and pad it to square,
    also additionally normalize the input
    """
    def __init__(self,
                 input_size : int,
                 scaler : Union[int,float] = 255,
                 mean : List[float] = [0.,0.,0.],
                 std : List[float] = [1.,1.,1.],
                 image_pad_value : Union[int,float] = 0,
                 labels_pad_value : Union[int,float] = -99,
                 normalize : bool = True):
        """Initialization

        Args:
            input_size (int): Target size of image resize
            scaler (Union[int,float], optional): The scaling factor applied to the input pixel value. Defaults to 255.
            mean (List[float], optional): Mean pixel values for image normalization. Defaults to [0.,0.,0.].
            std (List[float], optional): Standard deviation values for image normalization. Defaults to [1.,1.,1.].
            image_pad_value (Union[int,float], optional): Values of the color to pad the image to square.. Defaults to 0.
            labels_pad_value (Union[int,float], optional): Values used to pad the labels information so it have same dimension. Will be deleted on the dataloader. Defaults to -99.
            normalize (bool, optional): Will apply normalization if set to True. Defaults to True.
        """

        # By default, CropMirrorNormalize divide each pixel by 255, to make it similar with Pytorch Loader behavior
        # in which we can control the scaler, we add additional scaler to reverse the effect
        self.normalize = normalize
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
        """This function will receive keyword args which will be processed as a dict.
        Inside the dict exist 4 keys():
            - `images` : DataNode containing images data
            - `labels` : dictionary which corresponds to dataset.dataformat keys, which 
                         contain DataNode for each labels data
            - `additional_info` : dictionary which contain additional_info that also need to 
                                  be propagated out of the pipeline
            - `data_layout` : list containing the sequence of pipeline output names, if there is any
                              additional info want to be delivered out of the pipeline, it's name must
                              be registered here

        Returned augmentation result must  have the same format as mentioned above
        """
        data = EasyDict(data)
        images = data.images
        
        # Resize to model input size
        images = self.model_input_resize(images)
        
        # Save original shape before padding to fix label's with coordinates
        pre_padded_image_shapes = self.peek_shape(images)

        # Pad to square
        images = self.image_pad(images)
        
        # Normalize input if specified,
        if self.normalize:
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
    """Flip image in horizontal axis. Supports coordinates sensitive labels
    """
    def __init__(self,
                 p : float =0.5):
        """Initialization

        Args:
            p (float, optional): Probability to apply this transformation. Defaults to 0.5.
        """
        self.flip_coin_hflip = ops.CoinFlip(probability=p)
        self.image_hflip = ops.Flip(device='gpu')
        self.bbox_hflip = ops.BbFlip(device='cpu')
        self.ldmrks_hflip = ops.CoordFlip(layout='xy',device='cpu')

    def __call__(self,**data):
        """This function will receive keyword args which will be processed as a dict.
        Inside the dict exist 4 keys():
            - `images` : DataNode containing images data
            - `labels` : dictionary which corresponds to dataset.dataformat keys, which 
                         contain DataNode for each labels data
            - `additional_info` : dictionary which contain additional_info that also need to 
                                  be propagated out of the pipeline
            - `data_layout` : list containing the sequence of pipeline output names, if there is any
                              additional info want to be delivered out of the pipeline, it's name must
                              be registered here

        Returned augmentation result must  have the same format as mentioned above
        """
        
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
    """Flip image in vertical axis. Supports coordinates sensitive labels
    """
    def __init__(self,
                 p : float =0.5):
        """Initialization

        Args:
            p (float, optional): Probability to apply this transformation. Defaults to 0.5.
        """
        self.flip_coin_vflip = ops.CoinFlip(probability=p)
        self.image_vflip = ops.Flip(device='gpu',horizontal=0)
        self.bbox_vflip = ops.BbFlip(device='cpu',horizontal=0)
        self.ldmrks_vflip = ops.CoordFlip(layout='xy',device='cpu',flip_x=0)

    def __call__(self,**data):
        """This function will receive keyword args which will be processed as a dict.
        Inside the dict exist 4 keys():
            - `images` : DataNode containing images data
            - `labels` : dictionary which corresponds to dataset.dataformat keys, which 
                         contain DataNode for each labels data
            - `additional_info` : dictionary which contain additional_info that also need to 
                                  be propagated out of the pipeline
            - `data_layout` : list containing the sequence of pipeline output names, if there is any
                              additional info want to be delivered out of the pipeline, it's name must
                              be registered here

        Returned augmentation result must  have the same format as mentioned above
        """
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
    """Randomly adjust the brightness and contrast of the image
    """
    def __init__(self,
                 p : float = .5,
                 brightness_limit : Union[List,float] = 0.5,
                 contrast_limit : Union[List,float] = 0.5):
        """Initialization

        Args:
            p (float, optional): Probability to apply this transformation. Defaults to .5.
            brightness_limit (Union[List,float], optional): Factor multiplier range for changing brightness in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5.
            contrast_limit (Union[List,float], optional): Factor multiplier range for changing contrast in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5.
        """
        brightness_limit = _check_and_convert_limit_value(brightness_limit)
        contrast_limit = _check_and_convert_limit_value(contrast_limit)
        self.brightness_uniform = ops.Uniform(range=brightness_limit)
        self.contrast_uniform = ops.Uniform(range=contrast_limit)
        self.random_brightness_contrast = ops.BrightnessContrast(device='gpu')

        self.rng = ops.CoinFlip(probability = p)
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)

    def __call__(self,**data):
        """This function will receive keyword args which will be processed as a dict.
        Inside the dict exist 4 keys():
            - `images` : DataNode containing images data
            - `labels` : dictionary which corresponds to dataset.dataformat keys, which 
                         contain DataNode for each labels data
            - `additional_info` : dictionary which contain additional_info that also need to 
                                  be propagated out of the pipeline
            - `data_layout` : list containing the sequence of pipeline output names, if there is any
                              additional info want to be delivered out of the pipeline, it's name must
                              be registered here

        Returned augmentation result must  have the same format as mentioned above
        """
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
    """Perform a random Jitter augmentation. The output image is produced by moving each pixel 
    by a random amount bounded by half of nDegree parameter (in both x and y dimensions).
    """

    def __init__(self,
                 p : float = .5,
                 nDegree : int = 2,
                 fill_value : Union[float,int] = 0.):
        """Initialization

        Args:
            p (float, optional): Probability to apply this transformation. Defaults to .5.
            nDegree (int, optional): Each pixel is moved by a random amount in range [-nDegree/2, nDegree/2]. Defaults to 2.
            fill_value (Union[float,int], optional): Color value used for padding pixels. Defaults to 0..
        """

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
    """ Randomly performs HSV manipulation. To change hue, saturation and/or value of the image, pass corresponding coefficients. Keep in mind, that hue has additive delta argument, while for saturation and value they are multiplicative.
    """
    def __init__(self,
                 p : float = .5,
                 hue_limit : Union[List,float] = 20.,
                 saturation_limit : Union[List,float] = .5,
                 value_limit : Union[List,float] = .5):
        """Initialization

        Args:
            p (float, optional): Probability to apply this transformation. Defaults to .5.
            hue_limit (Union[List,float], optional): Range for changing hue in [min,max] value format. If provided as a single float, the range will be (-limit, limit). Defaults to 20..
            saturation_limit (Union[List,float], optional): Factor multiplier range for changing saturation in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5.
            value_limit (Union[List,float], optional): Factor multiplier range for changing value in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5.
        """
        
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
        """This function will receive keyword args which will be processed as a dict.
        Inside the dict exist 4 keys():
            - `images` : DataNode containing images data
            - `labels` : dictionary which corresponds to dataset.dataformat keys, which 
                         contain DataNode for each labels data
            - `additional_info` : dictionary which contain additional_info that also need to 
                                  be propagated out of the pipeline
            - `data_layout` : list containing the sequence of pipeline output names, if there is any
                              additional info want to be delivered out of the pipeline, it's name must
                              be registered here

        Returned augmentation result must  have the same format as mentioned above
        """
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

class RandomWater():
    """Randomly perform a water augmentation (make image appear to be underwater)
    Currently not support coordinates sensitive labels
    """

    def __init__(self,
                 p : float = .5,
                 ampl_x : float =10.0,
                 ampl_y : float =10.0,
                 freq_x : float =0.049087,
                 freq_y : float =0.049087,
                 phase_x : float =0.0,
                 phase_y : float =0.0,
                 fill_value : float =0.0):
        """Initialization

        Args:
            p (float, optional): Probability to apply this transformation. Defaults to .5.
            ampl_x (float, optional): Amplitude of the wave in x direction.. Defaults to 10.0.
            ampl_y (float, optional): Amplitude of the wave in y direction.. Defaults to 10.0.
            freq_x (float, optional): Frequency of the wave in x direction. Defaults to 0.049087.
            freq_y (float, optional): Frequency of the wave in y direction. Defaults to 0.049087.
            phase_x (float, optional): Phase of the wave in x direction.. Defaults to 0.0.
            phase_y (float, optional): Phase of the wave in y direction.. Defaults to 0.0.
            fill_value (float, optional): Color value used for padding pixels. Defaults to 0.0.
        """
        
        self.water_aug = ops.Water(device='gpu',
                                   ampl_x=ampl_x,
                                   ampl_y=ampl_y,
                                   freq_x=freq_x,
                                   freq_y=freq_y,
                                   phase_x=phase_x,
                                   phase_y=phase_y,
                                   fill_value=fill_value
                                   )
        self.rng = ops.CoinFlip(probability = p)
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)

    def __call__(self,**data):
        """This function will receive keyword args which will be processed as a dict.
        Inside the dict exist 4 keys():
            - `images` : DataNode containing images data
            - `labels` : dictionary which corresponds to dataset.dataformat keys, which 
                         contain DataNode for each labels data
            - `additional_info` : dictionary which contain additional_info that also need to 
                                  be propagated out of the pipeline
            - `data_layout` : list containing the sequence of pipeline output names, if there is any
                              additional info want to be delivered out of the pipeline, it's name must
                              be registered here

        Returned augmentation result must  have the same format as mentioned above
        """
        data = EasyDict(data)

        aug_images = self.water_aug(data.images)

        # DALI multiplexing to apply probability for applied augmentation or not
        apply_condition = self.bool(self.rng())
        neg_condition = apply_condition ^ True
        data.images = apply_condition * aug_images + neg_condition * data.images

        return data

class RandomRotate():
    """Random rotate the image, currently not support coordinates sensitive labels
    """

    def __init__(self,
                 p : float = .5,
                 angle_limit: Union[List,float] = 45.,
                 fill_value : float = 0.):
        """Initialization

        Args:
            p (float, optional): Probability to apply this transformation.. Defaults to .5.
            angle_limit (Union[List,float], optional): Range for changing angle in [min,max] value format. If provided as a single float, the range will be (-limit, limit). Defaults to 45..
            fill_value (float, optional): [description]. Defaults to 0..
        """

        angle_limit = _check_and_convert_limit_value(angle_limit,None,0)

        self.angle_uniform = ops.Uniform(range=angle_limit)

        self.rotate = ops.Rotate(device='gpu',
                                 fill_value = 0.,
                                 keep_size=True)

        self.rng = ops.CoinFlip(probability = p)
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)

    def __call__(self,**data):
        """This function will receive keyword args which will be processed as a dict.
        Inside the dict exist 4 keys():
            - `images` : DataNode containing images data
            - `labels` : dictionary which corresponds to dataset.dataformat keys, which 
                         contain DataNode for each labels data
            - `additional_info` : dictionary which contain additional_info that also need to 
                                  be propagated out of the pipeline
            - `data_layout` : list containing the sequence of pipeline output names, if there is any
                              additional info want to be delivered out of the pipeline, it's name must
                              be registered here

        Returned augmentation result must  have the same format as mentioned above
        """
        data = EasyDict(data)

        angle_rng = self.angle_uniform()
        aug_images = self.rotate(data.images,
                                  angle = angle_rng)

        # DALI multiplexing to apply probability for applied augmentation or not
        apply_condition = self.bool(self.rng())
        neg_condition = apply_condition ^ True
        data.images = apply_condition * aug_images + neg_condition * data.images

        return data

