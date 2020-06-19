## Experiments

| Experiment File | Backbone | Head | Loss  | Train Dataset    | Optimizer & Scheduler | Pretrained  | Validation Result | Notes |   
| --------------- | :------: | :--: | :---: | :--------------: | :-------------------: | :---------: | ----------------- | ----- |
| [efficientnet_b0_classification_stl10_224.yml](efficientnet_b0_classification_stl10_224.yml) | `efficientnet_b0` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.962125` | 20 epoch, image res 224 |
| [efficientnet_lite0_classification_stl10_224.yml](efficientnet_lite0_classification_stl10_224.yml) | `efficientnet_lite0` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.9305` | 20 epoch, image res 224 |
| [mobilenetv2_classification_stl10_224.yml](mobilenetv2_classification_stl10_224.yml) | `mobilenetv2` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.94475` | 20 epoch, image res 224 |
| [resnet18_classification_stl10_224.yml](resnet18_classification_stl10_224.yml) | `resnet18` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.953125` | 20 epoch, image res 224 |
| [shufflenetv2x100_classification_stl10_224.yml](shufflenetv2x100_classification_stl10_224.yml) | `shufflenetv2x100` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.920625` | 20 epoch, image res 224 |
| [mobilenetv2_classification_cifar10.yml](mobilenetv2_classification_cifar10.yml) | `mobilenetv2` | `softmax` |  `CE` | `CIFAR10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8401` | 20 epoch, image res 32 |
| [resnet18_classification_cifar10.yml](resnet18_classification_cifar10.yml) | `resnet18` | `softmax` |  `CE` | `CIFAR10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8451` | 20 epoch, image res 32 |
| [shufflenetv2x100_classification_cifar10.yml](shufflenetv2x100_classification_cifar10.yml) | `shufflenetv2x100` | `softmax` |  `CE` | `CIFAR10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.7125` | 20 epoch, image res 32 |
| [efficientnet_lite0_classification_cifar10.yml](efficientnet_lite0_classification_cifar10.yml) | `efficientnet_lite0` | `softmax` |  `CE` | `CIFAR10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8251` | 20 epoch, image res 32 |
| [efficientnet_b0_classification_cifar10.yml](efficientnet_b0_classification_cifar10.yml) | `efficientnet_b0` | `softmax` |  `CE` | `CIFAR10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8278` | 20 epoch, image res 32 |
| [mobilenetv2_classification_stl10.yml](mobilenetv2_classification_stl10.yml) | `mobilenetv2` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.88` | 20 epoch, image res 96 |
| [resnet18_classification_stl10.yml](resnet18_classification_stl10.yml) | `resnet18` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.88125` | 20 epoch, image res 96 |
| [shufflenetv2x100_classification_stl10.yml](shufflenetv2x100_classification_stl10.yml) | `shufflenetv2x100` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8575` | 20 epoch, image res 96 |
| [efficientnet_lite0_classification_stl10.yml](efficientnet_lite0_classification_stl10.yml) | `efficientnet_lite0` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.861625` | 20 epoch, image res 96 |
| [efficientnet_b0_classification_stl10.yml](efficientnet_b0_classification_stl10.yml) | `efficientnet_b0` | `softmax` |  `CE` | `STL10` | `SGD` & `CosineLR` | dvc | `accuracy: 0.895875` | 20 epoch, image res 96 |
| [mobilenetv2_classification_cifar100.yml](mobilenetv2_classification_cifar100.yml) | `mobilenetv2` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.6128` | 50 epoch, image res 32 |
| [resnet18_classification_cifar100.yml](resnet18_classification_cifar100.yml) | `resnet18` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.6072` | 50 epoch, image res 32 |
| [shufflenetv2x100_classification_cifar100.yml](shufflenetv2x100_classification_cifar100.yml) | `shufflenetv2x100` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.5237` | 10 epoch, image res 32 |
| [efficientnet_lite0_classification_cifar100.yml](efficientnet_lite0_classification_cifar100.yml) | `efficientnet_lite0` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.597` | 50 epoch, image res 32 |
| [efficientnet_b0_classification_cifar100.yml](efficientnet_b0_classification_cifar100.yml) | `efficientnet_b0` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.6117` | 50 epoch, image res 32 |
| [mobilenetv2_classification_cifar100_224.yml](mobilenetv2_classification_cifar100_224.yml) | `mobilenetv2` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.797` | 10 epoch, image res 224 |
| [resnet18_classification_cifar100_224.yml](resnet18_classification_cifar100_224.yml) | `resnet18` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8116` | 10 epoch, image res 224 |
| [shufflenetv2x100_classification_cifar100_224.yml](shufflenetv2x100_classification_cifar100_224.yml) | `shufflenetv2x100` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.7898` | 10 epoch, image res 224 |
| [efficientnet_lite0_classification_cifar100_224.yml](efficientnet_lite0_classification_cifar100_224.yml) | `efficientnet_lite0` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8353` | 10 epoch, image res 224 |
| [efficientnet_b0_classification_cifar100_224.yml](efficientnet_b0_classification_cifar100_224.yml) | `efficientnet_b0` | `softmax` |  `CE` | `CIFAR100` | `SGD` & `CosineLR` | dvc | `accuracy: 0.8698` | 10 epoch, image res 224 |

## Example Results

| [efficientnet_b0_classification_stl10_224.yml](efficientnet_b0_classification_stl10_224.yml) Confusion Matrix |
| --------------- |
| ![efficientnet_b0_softmax_stl10_224_cuda:0_BasePredictor[cuda:0].png](../outputs/efficientnet_b0_softmax_stl10_224/efficientnet_b0_softmax_stl10_224_cuda:0_BasePredictor[cuda:0].png) |