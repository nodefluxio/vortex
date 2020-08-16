## Vortex Custom Model Example: AlexNet
Demonstrate steps to create custom classification model and integrate to vortex.

### Files
- `alexnet.py`: custom model implementation, as well as experiment entrypoint
- `cifar10.yml`: configuration file for train, validate, on CIFAR10 as well as onnx export
- `hyperparam.yml`: configuration file for hyperparameter optimization, will optimize learning rates and scheduler

### Usage
- Hyperparameter Optimization
  ```
  python3 alexnet.py hypopt --config cifar10.yml --optconfig hyperparam.yml
  ```
- Training
  ```
  python3 alexnet.py train --config cifar10.yml
  ```
- TorchScript & ONNX Export
  ```
  python3 alexnet.py export --config cifar10.yml
  ```
- TensorRT Inference (using exported ONNX model)
  ```
  python3 alexnet.py predict --config cifar10.yml -i example_image.jpg
  ```
