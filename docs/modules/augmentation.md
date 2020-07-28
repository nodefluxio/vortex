# Augmentations

This section listed all available `augmentations` modules configurations. Part of [main configurations](../user-guides/experiment_file_config.md#dataset) in experiment file.

---

## Albumentations

You can utilized integrated [albumentations](https://albumentations.readthedocs.io/en/latest/) image augmentation library in Vortex.

Originally users must code the following example to use it. E.g. :

```python
from albumentations.augmentations.transforms import (
    HorizontalFlip,
    RandomScale,
    RandomBrightnessContrast,
    RandomSnow
)
from albumentations.core.composition import (
    Compose,
    OneOf,
    BboxParams
)

bbox_params=BboxParams(format='coco',min_area=0.0, min_visibility=0.0)

transforms=Compose([OneOf(transforms=[RandomBrightnessContrast(p=0.5),
                                      RandomSnow(p=0.5)],p=0.5),
                    HorizontalFlip(p=0.5),
                    RandomScale(p=0.5,scale_limit=0.1)],
                   bbox_params=bbox_params)
```

In Vortex we simplify it in the experiment file, and abstracted the process of each image and targets. So the analogous form of the above script in Vortex configuration is shown below. E.g. :

```yaml
augmentations: [
    {
    module: albumentations,
    args: {
        transforms: [
            { compose: OneOf, args: {
                transforms: [
                    { transform: RandomBrightnessContrast, args: { p: 0.5}},
                    { transform: RandomSnow, args: { p: 0.5}}
             ],
                p: 0.5}
            },
            { transform: HorizontalFlip, args: { p: 0.5 } },
            { transform: RandomScale, args: { scale_limit: 0.1, p: 0.5 } }
        ],
        bbox_params: {
            min_visibility: 0.0,
            min_area: 0.0
        },
        visual_debug: False
    }
    }
]
```

Arguments :

- `transforms` (list[dict]) : list of augmentation transformation or compose to be sequentially added. Each member of the list is a dictionary with a sub-arguments shown below :

    - `compose` OR `transform` (str) : denotes a compose or a transform from albumentation. 
        
        - Only support [`OneOf` compose](https://albumentations.readthedocs.io/en/latest/api/core.html#albumentations.core.composition.OneOf) for `compose` key. 
        
        - Supported `transform` available in [this link](https://albumentations.readthedocs.io/en/latest/api/augmentations.html). 

    - `args` : the corresponding arguments for selected `compose` or `transform`

        - for `OneOf` compose, the `transforms` arguments have same description with the previous `transforms` (list[dict]). Possible for nested `compose` inside `transforms`

- `bbox_params` (dict) : Parameter for bounding box target's annotation. See [this link](https://albumentations.readthedocs.io/en/latest/api/core.html#albumentations.core.composition.BboxParams) for further reading. Supported sub-args:
    
    - `min_visibility` (float) : minimum fraction of area for a bounding box to remain this box in list
    - `min_area` (float) : minimum area of a bounding box. All bounding boxes whose visible area in pixels is less than this value will be removed

- `visual_debug` (bool) : used for visualization debugging. It uses ‘cv2.imshow’ to visualize every augmentations result. Disable it for training, default `False`

---

## Nvidia DALI

You can utilized integrated [Nvidia DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) image augmentation library in Vortex. However, due to the low level nature of DALI's ops, we provide several high level transformation module that automatically handle label's transformation for several ops that's label sensitive.

Similar like [`albumentations`](#albumentations) module, you can utilize Nvidia DALI augmentation by specifying the following configurations

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: HorizontalFlip, args: { p: 0.5 } },
            { transform: RandomBrightnessContrast, args: { brightness_limit: 0.2, contrast_limit : 0.1, p: 0.5 } }
        ]
    }
    }
]
```

Arguments :

- `transforms` (list[dict]) : list of augmentation transformation or compose to be sequentially added. Each member of the list is a dictionary with a sub-arguments shown below :

    - `transform` (str) : denotes a transform function to be used. Supported transforms can be read in the following section
    - `args` (dict) : the corresponding arguments for selected `transform`

### Supported Augmentations

The supported Nvidia DALI transforms listed in here :

#### Horizontal Flip

Flip image in horizontal axis. Supports coordinates sensitive labels

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: HorizontalFlip, args: { p: 0.5 } },
        ]
    }
    }
]
```

Arguments : 

- `p` (float) : probability of applying the transform. Default: 0.5.

#### Vertical Flip

Flip image in vertical axis. Supports coordinates sensitive labels

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: VerticalFlip, args: { p: 0.5 } },
        ]
    }
    }
]
```

Arguments : 

- `p` (float) : probability of applying the transform. Default: 0.5.

#### Random Brightness Contrast

Apply random brightness and contrast adjustment of the image

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: RandomBrightnessContrast, args: { p: 0.5 ,
                                                           brightness_limit: 0.5,
                                                           contrast_limit: 0.5
                                                         } },
        ]
    }
    }
]
```

Arguments : 

- `p` (float) : probability of applying the transform. Default: 0.5.
- `brightness_limit` (float,list) : Factor multiplier range for changing brightness in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5. 
- `contrast_limit` (float,list) : Factor multiplier range for changing contrast in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5.

#### Random Jitter

Perform a random Jitter augmentation. The output image is produced by moving each pixel by a random amount bounded by half of `nDegree` parameter (in both x and y dimensions).

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: RandomJitter, args: { p: 0.5 ,
                                               nDegree: 2,
                                             } },
        ]
    }
    }
]
```

Arguments : 

- `p` (float) : probability of applying the transform. Default: 0.5.
- `nDegree` (int) : Each pixel is moved by a random amount in range [-nDegree/2, nDegree/2]. Defaults to 2.

#### Random Hue Saturation Value
Apply random HSV manipulation. To change hue, saturation and/or value of the image, pass corresponding coefficients. Keep in mind, that hue has additive delta argument, while for saturation and value they are multiplicative.

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: RandomHueSaturationValue, args: { p: 0.5 ,
                                                           hue_limit: 20,
                                                           saturation_limit: 0.5,
                                                           value_limit: 0.5
                                                         } },
        ]
    }
    }
]
```

Arguments : 

- `p` (float) : probability of applying the transform. Default: 0.5.
- `hue_limit` (float,list) : Range for changing hue in [min,max] value format. If provided as a single float, the range will be (-limit, limit). Defaults to 20..
- `saturation_limit` (float,list) : Factor multiplier range for changing saturation in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5.
- `value_limit` (float,list) : Factor multiplier range for changing value in [min,max] value format. If provided as a single float, the range will be 1 + (-limit, limit). Defaults to 0.5.

#### Random Water
Apply random water augmentation (make image appear to be underwater). Can not support coordinates sensitive labels

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: RandomWater, args: { p: 0.5 ,
                                              ampl_x: 10.0,
                                              ampl_y: 10.0,
                                              freq_x: 0.049087,
                                              freq_y: 0.049087,
                                              phase_x: 0.0,
                                              phase_y: 0.0,
                                            } },
        ]
    }
    }
]
```

Arguments : 

- `p` (float) : probability of applying the transform. Default: 0.5.
- `ampl_x` (float): Amplitude of the wave in x direction. Defaults to 10.0.
- `ampl_y` (float): Amplitude of the wave in y direction. Defaults to 10.0.
- `freq_x` (float): Frequency of the wave in x direction. Defaults to 0.049087.
- `freq_y` (float): Frequency of the wave in y direction. Defaults to 0.049087.
- `phase_x` (float): Phase of the wave in x direction. Defaults to 0.0.
- `phase_y` (float): Phase of the wave in y direction. Defaults to 0.0.

#### Random Rotate
Apply random rotation to the image, currently not support coordinates sensitive labels

```yaml
augmentations: [
    {
    module: nvidia_dali,
    args: {
        transforms: [
            { transform: RandomRotate, args: { p: 0.5 ,
                                               angle_limit: 45.,
                                            } },
        ]
    }
    }
]
```

Arguments : 

- `p` (float) : probability of applying the transform. Default: 0.5.
- `angle_limit` (float,list) : Range for changing angle in [min,max] value format. If provided as a single float, the range will be (-limit, limit). Defaults to 45..