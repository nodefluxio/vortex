# Data Loader

This section listed all available `dataloader` modules configurations. Part of [`config.dataloader` configurations](../user-guides/experiment_file_config.md#dataset) in experiment file.

Vortex data loader module will normalize and resize the input image following `preprocess_args` configuration in [`model`](../user-guides/experiment_file_config.md#model) section. The resize function, will resize the image by the value provided in `input_size` configuration and pad the image to square by adding black pixel (0,0,0). However the padding implementation will be different depends on the dataloader module (this is subject to change in the future)

---

## Pytorch Data Loader

A standard Pytorch data loader. If the provided image data is image path, it will decode the image using OpenCV in the BGR format 

This data loader will pad the image in the shortest side by adding pad pixel in both edge. (E.g. if the shortest side is image's width, it will pad pixel in the left and right side of the image). The implementation is provided by `albumentations` augmentation [`PadIfNeeded`](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.PadIfNeeded)

Cannot be used with `nvidia_dali` augmentation module

```yaml
dataloader: {
    dataloader: PytorchDataLoader,
    args: {
        num_workers: 0,
        batch_size: 16,
        shuffle: True,
    },
}
```

It’s important to be noted that argument that expect function as its input is not supported, detailed arguments can be inspected in [this page](https://pytorch.org/docs/stable/data.html)

Common used arguments :

- `num_workers` (int) : how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process
- `batch_size` (int) : how many samples per batch to load (default: 1)
- `shuffle` (bool) : set to True to have the data reshuffled at every epoch (default: False).

---

## Nvidia DALI Data Loader

A data loader used together with [Nvidia DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) pipeline and must utilize a GPU. The input data provided must be the path to the image (cannot accept numpy array), and it will decode the image on the GPU using [ops.ImageDecoder](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.ImageDecoder) into BGR format.

This data loader will pad the image in the right-bottom style. The implementation is provided by [ops.Paste](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.Paste)

Can be used with `nvidia_dali` augmentation module by specifying the module in the first order of [`augmentations`](../user-guides/experiment_file_config/#dataset) configuration list

```yaml
dataloader: {
    dataloader: DALIDataLoader,
    args: {
        num_thread: 1,
        device_id: 0,
        batch_size: 16,
        shuffle: True,
    },
}
```

Arguments :

- `num_thread`(int) : Number of CPU threads used by the Nvidia DALI pipeline. (default: 1)
- `device_id` (int) : Id of GPU used by the pipeline. (default: 0)
- `batch_size` (int) : how many samples per batch to load (default: 1)
- `shuffle` (bool) : set to True to have the data reshuffled at every epoch (default: False).
