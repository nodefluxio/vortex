import warnings
import torchvision.datasets
import inspect
import numpy as np

CLASSIFICATION_DATASET = [
    'MNIST',
    'FashionMNIST',
    'KMNIST',
    'EMNIST',
    'QMNIST',
    'ImageFolder',
    'CIFAR10',
    'CIFAR100',
    'SVHN',
    'STL10'
]

SUPPORTED_TORCHVISION_DATASETS = CLASSIFICATION_DATASET


class TorchvisionBaseDatasetWrapper():
    def __init__(self, dataset: str, dataset_args: dict):

        # Supported dataset checker
        if dataset not in SUPPORTED_TORCHVISION_DATASETS:
            raise RuntimeError(
                "Torchvision dataset '%s' is not supported, please read Vortex documentation for supported dataset!!" % dataset)

        # Raise warnings if 'transform' and 'target_transform' args is provided and discard them from dataset args
        if 'transform' in dataset_args:
            dataset_args.pop('transform')
            warnings.warn("'transform' argument is not supported in this implementation, to use augmentation please read Vortex documentation about data augmentation!!")
        if 'target_transform' in dataset_args:
            dataset_args.pop('target_transform')
            warnings.warn("'target_transform' argument is not supported in this implementation, to use augmentation please read Vortex documentation about data augmentation!!")

        # If dataset args support 'loader' params, supply loader with identity function so it return image file path
        dataset_class = getattr(torchvision.datasets, dataset)
        if 'loader' in inspect.signature(dataset_class).parameters:
            dataset_args['loader'] = lambda x: x
        self.dataset = dataset_class(**dataset_args)

        # Added data format for supported classification dataset
        if dataset in CLASSIFICATION_DATASET:
            if dataset=='SVHN':
                self.class_names=[0,1,2,3,4,5,6,7,8,9]
            else:
                self.class_names = self.dataset.classes
            self.data_format = {
                'class_label': None
            }

    def __getitem__(self, index):
        img, target = self.dataset[index]

        # Handle if img returned is not the provided path
        if not isinstance(img, str):
            img = np.array(img)

            # For dataset with grayscale image, convert 1-channel image to 3-channel image
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)

        # Convert target to numpy array
        if isinstance(target, int):
            target = np.array([target])
        return img, target

    def __len__(self):
        return len(self.dataset)


def create_torchvision_dataset(dataset: str, dataset_args: dict):
    dataset = TorchvisionBaseDatasetWrapper(dataset, dataset_args)
    return dataset
