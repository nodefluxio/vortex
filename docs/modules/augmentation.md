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