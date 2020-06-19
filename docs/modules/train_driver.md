# Training Driver

This section listed all available `driver` configuration. Part of [`trainer` configurations](../user-guides/experiment_file_config.md#trainer) in experiment file.

---

## Default Trainer

The basic training driver. A model will be forwarded batched inputs, loss will be calculated and back propagated.  This trainer supports gradient accumulation in which backpropagation gradient can be bigger than batch size.

For example :

Let's say `batch_size` in `dataloader` is `16`, `accumulation_step` is `4`. The gradient calculation will come from `batch_size` * `accumulation_step` which is equivalent to `16*4=64` . Thus even if with limited resource, training with simulated larger batch size is possible.

E.g. :

```yaml
driver: {
    module: DefaultTrainer,
    args: {
        accumulation_step: 4,
    }
}
```

Arguments :

- `accumulation_step` (int) : number of iterations before gradient is back propagated

