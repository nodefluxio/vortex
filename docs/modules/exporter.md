# Graph Exporter

This section listed all available `exporter` configuration. Part of [graph exporter section](../user-guides/experiment_file_config.md#graph-exporter) in experiment file.

---

## ONNX

This module will produce ONNX IR from a trained Vortex model. Further reading : [onnx.ai](https://onnx.ai/)

Currently, we provide export support for opset version 9,10, and 11. However it must be noted that not all models and backbones are supported to be converted to ONNX. The list of supported and not supported models and backbones can be found in the COMPATIBILITY REPORT on the repo :

- [COMPATIBILITY REPORT Opset Version 9](https://github.com/nodefluxio/vortex/blob/master/COMPATIBILITY_REPORT_opset9.md)
- [COMPATIBILITY REPORT Opset Version 10](https://github.com/nodefluxio/vortex/blob/master/COMPATIBILITY_REPORT_opset10.md)
- [COMPATIBILITY REPORT Opset Version 11](https://github.com/nodefluxio/vortex/blob/master/COMPATIBILITY_REPORT_opset11.md)

E.g. :

```yaml
exporter: {
    module: onnx,
    args: {
        n_batch: 4,
        opset_version: 11,
        filename: somemodel_bs4
    },
}
```

Arguments : 

- `n_batch` (int) : number of input batches that can be processed by this model at a time. An IR graph input batch size must be pre configured during export and cannot be modified afterwise
- `opset_version` (int) : selected ONNX opset version. For complete information check this link
- `filename` (str) : A new filename which will be given to the exported IR with `.onnx` suffix. If not given, the default is `{experiment_name}.onnx`

Outputs :

- ONNX IR : ONNX file model `(*.onnx)` will be produced at experiment output directory

---

## Torchscript

This module will produce Torchscript IR from a trained Vortex model. Further reading : [torchscript](https://pytorch.org/docs/stable/jit.html)

E.g. :

```yaml
exporter: {
    module: torchscript,
    args: {
        n_batch: 4,
        filename: somemodel_bs4
    },
}
```

Arguments : 

- `n_batch` (int) : number of input batches that can be processed by this model at a time. An IR graph input batch size must be pre configured during export and cannot be modified afterwise
- `filename` (str) : A new filename which will be given to the exported IR with `.pt` suffix. If not given, the default is `{experiment_name}.pt`

Outputs :

- Torhscript IR : Torchscript file model `(*.pt)` will be produced at experiment output directory