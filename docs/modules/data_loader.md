# Data Loader

This section listed all available `dataloader` modules configurations. Part of [`dataset.dataloader` configurations](../user-guides/experiment_file_config.md#dataset) in experiment file.

---

## Pytorch Data Loader

A standard Pytorch data loader.

```yaml
dataloader: {
    dataloader: DataLoader,
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

