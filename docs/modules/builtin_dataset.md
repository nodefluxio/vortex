# Built-in Dataset

In this module, we listed all the internally supported dataset to be used in Vortex. Part of [`dataset.train` AND `dataset.eval` configurations](../user-guides/experiment_file_config.md#dataset) in experiment file.

---

## Torchvision Dataset

Several [torchvision dataset](https://pytorch.org/docs/stable/torchvision/datasets.html) is supported, they are listed below :

- [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)
- [FashionMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)
- [KMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#kmnist)
- [EMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#emnist)
- [QMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#emnist)
- [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)
- [CIFAR10](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)
- [CIFAR100](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)
- [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn)
- [STL10](https://pytorch.org/docs/stable/torchvision/datasets.html#stl10)

To use these dataset, set the experiment file using the dataset identifier listed above, and pass the arguments like shown in their respective documentations in the `args` field. For example :

```yaml
dataset: {

    train: {
        dataset: CIFAR10,
        args: {
            root: external/datasets,
            train: True,
            download: True
        }
    },

    eval: {
        dataset: CIFAR10,
        args: {
            root: external/datasets,
            train: False,
            download: True
        }
    }
    
}
```

**IMPORTANT NOTES : `transform` and `target_transform` arguments is not supported, augmentation will be supported by Vortex built-in augmentation mechanism specified in [this step](../user-guides/experiment_file_config.md#dataset)



