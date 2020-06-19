# Models Zoo

This section listed all available `model` configuration. Part of [model section](../user-guides/experiment_file_config.md#model) in experiment file.

---

## Classification

This task is about predicting a category for an input image.

---

### Softmax

This classification model supports single-category multi-class classification objectives. The output will be a class with the highest prediction probability score (between 0 and 1)

Reference : 

- [Softmax and probabilities](https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#softmax-and-probabilities)

E.g. :

```yaml
model: {
    name: softmax,
    network_args: {
        backbone: shufflenetv2_x1.0,
        n_classes: 10,
        pretrained_backbone: True,
        freeze_backbone: False
    },[
    preprocess_args: {
        input_size: 224,
        input_normalization: {
            mean: [ 0.5, 0.5, 0.5 ],
            std: [ 0.5, 0.5, 0.5 ],
            scaler: 255
        }
    },
    loss_args: {
        loss: ce_loss,
        reduction: mean,
        additional_args: {}
    },
    postprocess_args: {},
    init_state_dict: somepath.pth
}
```

#### Arguments

- `preprocess_args` : 

    see [this section](../user-guides/experiment_file_config.md#model)

- `network_args` :

    - `backbone` (str) : backbones network name, supported backbones network is provided at [backbones network section](../modules/backbones.md)
    - `n_classes` (int) : number of classes
    - `pretrained_backbone` (bool) :  using the provided pretrained backbone for weight initialization (transfer learning)
    - `freeze_backbone `(bool) : freeze the backbone weight, if the backbone is frozen the weight in backbone model will not be updated during training

- `loss_args` :

    - `loss` (str) : classification loss to be used, options :

        - `ce_loss` - will use `nn.NLLLoss` from PyTorch, see [this documentations](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss) for details.
        - `focal_loss` - will use Vortex implementation of Focal loss, referenced from [this repo](https://github.com/clcarwin/focal_loss_pytorch)
    
    - `reduction` (str) : reduction of loss array, either `sum` or `mean`
    - `additional_args` (dict) : additional arguments to be forwarded to the loss function, options:

        - for `ce_loss`, see the [documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss) for additional arguments
        - for `focal_loss` , see [this link](https://arxiv.org/abs/1708.02002) for further details, sub-arguments :

            - `gamma` (float) : focusing parameter for modulating factor (1-p)
            - `alpha` (List[float]) : per-class weighting array.

                For example, if you have a classification task with `3` class, you need to provide the weight in an array of size 3 = `[0.1 ,0.2, 0.7]`

- `postprocess_args` :

    you can leave this field with empty dict `{}`

- `init_state_dict` : 

    see [this section](../user-guides/experiment_file_config.md#model)

#### Outputs

List of dictionary of `np.ndarray` pair, output key :

- `class_label` (`np.ndarray`) : array with size of 1, each column represents class label
- `class_confidence` (`np.ndarray`) : array with size of 1, each column represents class confidence

**NOTES** : row orders are consistent : `class_confidence[i]` is associated with `class_label[i]`

---

## Detection

This task is about predicting multiple objects location as pixel coordinates and its category from input image.

Detection models’ params and MAC/FLOPS comparison ( to compare which models is the lightest or heaviest ) can be found on this [spreadsheet link](https://docs.google.com/spreadsheets/d/1M18Bm08P983_-5diHXAmlUmHusjpHdYMmcN0FmrNS74/edit#gid=189749985)

---

### FPN-SSD

This implementation is basically RetinaNet without (not yet) focal loss implemented.

**DISCLAIMER : THIS MODEL IS NOT VERIFIED YET, MAY PRODUCE BAD RESULT**

Reference :

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

E.g. 

```yaml
model: {
    name: FPNSSD,
    preprocess_args: {
        input_size: 512,
        input_normalization: {
            mean: [ 0.5, 0.5, 0.5 ],
            std: [ 0.5, 0.5, 0.5 ],
            scaler: 255
        }
    },
    network_args: {
        backbone: shufflenetv2_x1.0,
        n_classes: 20,
        pyramid_channels: 256,
        aspect_ratios: [ 1, 2., 3. ],
        pretrained_backbone: True,
        freeze_backbone: False
    },
    loss_args: {
        neg_pos: 3,
        overlap_thresh: 0.5,
    },
    postprocess_args: {},
    init_state_dict: somepath.pth
}
```

#### Arguments

- `preprocess_args` : 

    see [this section](../user-guides/experiment_file_config.md#model)

- `network_args` :

    - `backbone` (str) : backbones network name, supported backbones network is provided at [backbones network section](../modules/backbones.md)
    - `n_classes` (int) : number of classes
    - `pretrained_backbone` (bool) : using the provided pretrained backbone for weight initialization (transfer learning)
    - `freeze_backbone `(bool) : freeze the backbone weight, if the backbone is frozen the weight in backbone model will not be updated during training
    - `pyramid_channels` (int) : number of channels for FPN top-down pyramid, default to 256, could be modified if necessary (e.g. for accuracy vs speed trade-off) (the lower the number, speed will increase)(the lower the number, speed will increase)
    - `aspect_ratios` (list) : aspect ratio for anchor box

- `loss_args` :

    - `neg_pos` (int) : negative (background) to positive (object) ratio for Hard Negative Mining
    - `overlap_thresh` (float) : minimum iou threshold to be considered as positive during training

- `postprocess_args` :

    you can leave this field with empty dict `{}`

- `init_state_dict` : 

    see [this section](../user-guides/experiment_file_config.md#model)

#### Outputs

List of dictionary of `np.ndarray` pair, output key :

- `bounding_box` (`np.ndarray`) : array with size `n_detections * 4`, each column on `x_min,y_min,x_max,y_max` format
- `class_label` (`np.ndarray`) : array with size of `n_detectionx * 1`, each column represents class label
- `class_confidence` (`np.ndarray`) : array with size of `n_detections * 1`, each column represents class confidence

**NOTES** : row orders are consistent : `bounding_box[i]` is associated with `class_label[i]`, etc..

---

## Detection (with Landmarks/Keypoints)

This task is about predicting multiple objects location as pixel coordinates, its category, and objects landmarks/keypoints in `[x,y]` coordinates from input image.

Detection models’ params and MAC/FLOPS comparison ( to compare which models is the lightest or heaviest ) can be found on this [spreadsheet link](https://docs.google.com/spreadsheets/d/1M18Bm08P983_-5diHXAmlUmHusjpHdYMmcN0FmrNS74/edit#gid=189749985)

---

### RetinaFace

This model perform 1 class object detection with 5 key points estimation, originally used for face detection and face landmarks prediction

**DISCLAIMER : THIS MODEL IS NOT VERIFIED YET, MAY PRODUCE BAD RESULT**

Reference : 

- [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)

E.g. :

```yaml
model: {
    name: RetinaFace,
    preprocess_args: {
        input_size: 640,
        input_normalization: {
            mean: [ 0.5, 0.5, 0.5 ],
            std: [ 0.5, 0.5, 0.5 ]
        }
    },
    network_args: {
        backbone: shufflenetv2_x1.0,
        pyramid_channels: 64,
        aspect_ratios: [ 1, 2., 3. ],
        pretrained_backbone: True,
        freeze_backbone: False
    },
    loss_args: {
        neg_pos: 7,
        overlap_thresh: 0.35,
        cls: 2.0,
        box: 1.0,
        ldm: 1.0,
    },
    postprocess_args: {},
    init_state_dict: somepath.pth
}
```

#### Arguments

- `preprocess_args` : 

    see [this section](../user-guides/experiment_file_config.md#model)

- `network_args` :

    - `backbone` (str) : backbones network name, supported backbones network is provided at [backbones network section](../modules/backbones.md)
    - `pretrained_backbone` (bool) : using the provided pretrained backbone for weight initialization (transfer learning)
    - `freeze_backbone `(bool) : freeze the backbone weight, if the backbone is frozen the weight in backbone model will not be updated during training
    - `pyramid_channels` (int) : number of channels for FPN top-down pyramid, default to 256, could be modified if necessary (e.g. for accuracy vs speed trade-off) (the lower the number, speed will increase)(the lower the number, speed will increase)
    - `aspect_ratios` (list) : aspect ratio for anchor box

- `loss_args` :

    - `neg_pos` (int) : negative (background) to positive (object) ratio for Hard Negative Mining
    - `overlap_thresh` (float) : minimum iou threshold to be considered as positive during training
    - `cls` (float) : weight for classification loss
    - `box` (float) : weight for bounding box loss
    - `ldm` (float) : weight for landmark regression loss

- `postprocess_args` :

    you can leave this field with empty dict `{}`

- `init_state_dict` : 

    see [this section](../user-guides/experiment_file_config.md#model)

#### Outputs

List of dictionary of `np.ndarray` pair, output key :

- `bounding_box` (`np.ndarray`) : array with size `n_detections * 4`, each column on `x_min,y_min,x_max,y_max` format
- `class_label` (`np.ndarray`) : array with size of `n_detectionx * 1`, each column represents class label
- `class_confidence` (`np.ndarray`) : array with size of `n_detections * 1`, each column represents class confidence
- `landmarks` (`np.ndarray`) : array with size of `n_detections x 10`, each column represents landmark position with `xy` format, e.g. `[p1x, p1y, p2x, p2y, …]`


**NOTES** : row orders are consistent : `bounding_box[i]` is associated with `class_label[i]`, etc..

---