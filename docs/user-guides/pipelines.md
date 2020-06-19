# Vortex Pipelines

This section will describe how to easily run each of Vortex pipeline in details. For complete pipelines flow please see [Vortex overview section](../index.md#overview)

---

## Training Pipeline

This pipeline purpose is to train a deep learning model using the provided dataset. If you need to integrate the training into your own script you can see the [training pipeline API section](../api/vortex.core.pipelines.md#trainingpipeline).

To run this pipeline, make sure you've already prepared :

- **Dataset** : see [this section](../modules/builtin_dataset.md) for built-in datasets, or [this section](dataset_integration.md) for external datasets
- **Experiment file** : see [this section](experiment_file_config.md) to create one

You only need to run this command from the command line interface :

```console
usage: vortex train [-h] -c CONFIG [--no-log]

Vortex training pipeline; will generate a Pytorch model file

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to experiment config file
  --no-log              disable logging, ignore experiment file config
```

E.g. :

```console
vortex train -c experiments/config/efficientnet_b0_classification_cifar10.yml
```

This pipeline will generate several outputs :

- **Local runs log file** : every time a user runs a VORTEX training experiment, the experiment logger module will write a local file `experiments/local_runs.log` which will record all experimental training runs which have already executed sequentially for easier tracking. Example of the content inside `experiments/local_runs.log` is shown below :

        ###############################################################################
        Timestamp : 03/27/2020, 09:24:00
        Experiment Name : test_torchvision_dataset
        Output Path : experiments/outputs/test_torchvision_dataset/601f45782a884286be310b1ffe562597
        Logging Provider : comet_ml
        Experiment Log URL : https://www.comet.ml/hyperion-rg/vortex-dev/601f45782a884286be310b1ffe562597
        ###############################################################################

- **Experiment directory** : If not exist yet, training script will make a directory under the configured `output_directory` path. The created directory will be named after the `experiment_name` configuration and will be the directory to dump training (final weight), validation result (if any), backup, etc.

- **Run directory** : Everytime user runs the training script, it will be tagged as a new experiment run. Vortex (or third party logger) will generate a unique key which will be an identifier for that specific experiment run. And thus, Vortex will make a new directory under the experiment directory which will act as a backup directory. It will store the duplicate of the executed experiment file (as a backup) and will be the directory which store intermediate model’s weight path (weight that saved every n-epoch). For example, in the previous example log the output path is :

        Output Path : experiments/outputs/test_torchvision_dataset/601f45782a884286be310b1ffe562597
    
    The experiment directory is `test_torchvision_dataset` and the run directory is `601f45782a884286be310b1ffe562597`

- **Backup experiment file** : Experiment file will be duplicated and stored under **run directory**
- **Intermediate model weight** : Model’s weight will be dumped into **run directory** every *n*-epoch which is configured in `save_epoch` in experiment file with `.pth` extension
- **Final model weight** : Model’s weight after all training epoch is completed will be dumped in the **experiment directory** with `.pth` extension
- **Experiment log** : If logging is enabled, training metrics will be collected by the logging provider. Additionally if the config file is valid for validation, the validation metrics will also be collected.

---

## Validation Pipeline

This pipeline handle the evaluation of the Vortex model (Pytorch state dict `.pth`) in term of model's performance and resource usage. In addition, this pipeline also generate a visual report. If you need to integrate the validation into your own script you can see the [validation pipeline API section](../api/vortex.core.pipelines.md#pytorchvalidationpipeline).

To run this pipeline, make sure you've already prepared :

- **Validation dataset** : see [this section](../modules/builtin_dataset.md) for built-in datasets, or [this section](dataset_integration.md) for external datasets
- **Experiment file** : see [this section](experiment_file_config.md) to create one. Must be valid for validation, make sure [`dataset.eval`](experiment_file_config.md#dataset) and [`trainer.validation`](experiment_file_config.md#trainer) is set
- **Vortex model's file** `*.pth` : obtained from [training pipeline](#training-pipeline) which corresponds to the previous mentioned experiment file

You only need to run this command from the command line interface :

```console
usage: vortex validate [-h] -c CONFIG [-w WEIGHTS] [-v] [--quiet]
                       [-d [DEVICES [DEVICES ...]]] [-b BATCH_SIZE]

Vortex Pytorch model validation pipeline; successful runs will produce
autogenerated reports

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to experiment config
  -w WEIGHTS, --weights WEIGHTS
                        path to selected weights(optional, will be inferred
                        from `output_directory` and `experiment_name` field
                        from config) if not specified
  -v, --verbose         verbose prediction output
  --quiet
  -d [DEVICES [DEVICES ...]], --devices [DEVICES [DEVICES ...]]
                        computation device to be used for prediction, possible
                        to list multiple devices
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size for validation
```

**NOTES** : if `--weights` is not provided, Vortex will assume final weights exist in the **experiment directory**


E.g. :

```console
vortex validate -c experiments/config/efficientnet_b0_classification_cifar10.yml \
                -b 8 \
                -d cpu cuda
```

This pipeline will generate several outputs :

- **Report file** : after successful evaluation, report file will be generated under directory `reports` in the **experiment directory** based on `experiment_name` under `output_directory`. Pro Tip : the generated report could be easily converted to pdf using [pandoc](https://pandoc.org/demos.html) or [vscode markdown-pdf extension](https://marketplace.visualstudio.com/items?itemName=yzane.markdown-pdf).

---

## Prediction Pipeline

This pipeline is used to test and visualize your Vortex model's prediction. If you need to integrate the prediction into your own script you can see the [prediction pipeline API section](../api/vortex.core.pipelines.md#pytorchpredictionpipeline).

To run this pipeline, make sure you've already prepared :

- **Experiment file** : see [this section](experiment_file_config.md) to create one
- **Vortex model's file** `*.pth` : obtained from [training pipeline](#training-pipeline) which corresponds to the previous mentioned experiment file
- **Input image(s)** : image file(s) (tested with `*.jpg`,`*.jpeg`,`*.png` extension)

You only need to run this command from the command line interface :

```console
usage: vortex predict [-h] -c CONFIG [-w WEIGHTS] [-o OUTPUT_DIR] -i IMAGE
                      [IMAGE ...] [-d DEVICE]
                      [--score_threshold SCORE_THRESHOLD]
                      [--iou_threshold IOU_THRESHOLD]

Vortex Pytorch model prediction pipeline; may receive multiple image(s) for
batched prediction

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to experiment config
  -w WEIGHTS, --weights WEIGHTS
                        path to selected weights(optional, will be inferred
                        from `output_directory` and `experiment_name` field
                        from config) if not specified
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        directory to dump prediction visualization
  -i IMAGE [IMAGE ...], --image IMAGE [IMAGE ...]
                        path to test image(s)
  -d DEVICE, --device DEVICE
                        the device in which the inference will be performed
  --score_threshold SCORE_THRESHOLD
                        score threshold for detection, only used if model is
                        detection, ignored otherwise
  --iou_threshold IOU_THRESHOLD
                        iou threshold for nms, , only used if model is
                        detection, ignored otherwise
```

**NOTES** : if `--weights` is not provided, Vortex will assume final weights exist in the **experiment directory**

E.g. :

```console
vortex predict -c experiments/config/efficientnet_b0_classification_cifar10.yml \
               -i image1.jpg image2.jpg \
               -d cuda \
               -o output_vis
```

**NOTES** : Provided multiple input images will be treated as batch input

This pipeline will generate several outputs :

- **Output Visualization Directory** : if `--output_dir` is provided, it will create the directory in your current working directory
- **Prediction Visualization** : prediction visualization will be generated in the `--output_dir` if provided, or in the current working dir if not. The generated file will have `prediction_` name prefix.

---

## Hyperparameters Optimization Pipeline

This pipeline is used to search for optimum hyperparameter to be used for either training pipeline or validation pipeline (parameter in validation pipeline also can be used for prediction pipeline). Basically this pipeline is [Optuna](https://optuna.org/) wrapper for Vortex components. If you need to integrate the prediction into your own script you can see the [hyperparameters optimization pipeline API section](../api/vortex.core.pipelines.md#hypoptpipeline).

To run this pipeline, make sure you've already prepared :

- **Hypopt config file** : see [this section](hypopt_file_config.md) to create one
- **Experiment file** : see [this section](experiment_file_config.md) to create one
- **Vortex model's file** `*.pth` : obtained from [training pipeline](#training-pipeline) which corresponds to the previous mentioned experiment file
- Related objective's pipeline requirement

You only need to run this command from the command line interface :

```console
usage: vortex hypopt [-h] -c CONFIG -o OPTCONFIG [-w WEIGHTS]

Vortex hyperparameter optimization experiment

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to experiment config file
  -o OPTCONFIG, --optconfig OPTCONFIG
                        path to hypopt config file
  -w WEIGHTS, --weights WEIGHTS
                        path to selected weights (optional, will be inferred
                        from `output_directory` and `experiment_name` field
                        from config) if not specified, valid only for
                        ValidationObjective, ignored otherwise
```

**NOTES** : if `--weights` is not provided, Vortex will assume final weights exist in the **experiment directory**

E.g. :

```console
vortex hypopt -c experiments/config/efficientnet_b0_classification_cifar10.yml \
              -o experiments/hypopt/learning_rate_search.yml
```

This pipeline will generate several outputs :

- **Hypopt Output Dir** : `hypopt/{hypopt_study_name}` will be created under **experiment directory**
- **Best Parameters** : file `*.txt` containing best parameters will be created in **hypopt output dir**
- **Hypopt Visualization** : graph visualization of parameter search (visualization extension must be installed. see [installation section](../index.md#installation)) will be created in **hypopt output dir**

---

## Graph Export Pipeline

This pipeline is used to export trained Vortex model (or graph) into another graph representation (or Intermediate Representation (IR)). If you need to integrate the graph export pipeline into your own script you can see the [graph export pipeline API section](../api/vortex.core.pipelines.md#graphexportpipeline).

To run this pipeline, make sure you've already prepared :

- **Experiment file** : see [this section](experiment_file_config.md) to create one and make sure the `exporter` section is already configured
- **Vortex model's file** `*.pth` : obtained from [training pipeline](#training-pipeline) which corresponds to the previous mentioned experiment file
- **Example Input Image** : example input image for correct graph tracing. Recommended for using image from training dataset and strongly recommended for model with detection task

You only need to run this command from the command line interface :

```console
usage: vortex export [-h] -c CONFIG [-w WEIGHTS] [-i EXAMPLE_INPUT]

export model to specific IR specified in config, output IR are stored in the
experiment directory based on `experiment_name` under `output_directory`
config field, after successful export, you should be able to visualize the
network using [netron](https://lutzroeder.github.io/netron/)

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        export experiment config file
  -w WEIGHTS, --weights WEIGHTS
                        path to selected weights (optional, will be inferred
                        from `output_directory` and `experiment_name` field
                        from config) if not specified
  -i EXAMPLE_INPUT, --example-input EXAMPLE_INPUT
                        path to example input for tracing (optional, may be
                        necessary for correct tracing, especially for
                        detection model)
```

**NOTES** : if `--weights` is not provided, Vortex will assume final weights exist in the **experiment directory**

E.g. :

```console
vortex export -c experiments/config/efficientnet_b0_classification_cifar10.yml -i image1.jpg
```

This pipeline will generate several outputs :

- **IR model file** : IR model file will be created under **experiment directory**, with file extension that correspond to [`exporter`](../modules/exporter.md) settings

---

## IR Validation Pipeline

This pipeline handle the evaluation of the IR model (`*.pt` or `*.onnx`) in term of model's performance and resource usage. In addition, this pipeline also generate a visual report. If you need to integrate the validation into your own script you can see the [IR validation pipeline API section](../api/vortex.core.pipelines.md#irvalidationpipeline).

To run this pipeline, make sure you've already prepared :

- **Validation dataset** : see [this section](../modules/builtin_dataset.md) for built-in datasets, or [this section](dataset_integration.md) for external datasets
- **Experiment file** : see [this section](experiment_file_config.md) to create one. Must be valid for validation, make sure [`dataset.eval`](experiment_file_config.md#dataset) and [`trainer.validation`](experiment_file_config.md#trainer) is set
- **IR model file** `*.pt` or `*.onnx` : obtained from [graph export pipeline](#graph-export-pipeline)
- **IR runtime library and environment** : make sure runtime library and environment is installed (currently runtime library installed together with vortex)

You only need to run this command from the command line interface :

```console
usage: vortex ir_runtime_validate [-h] -c CONFIG -m MODEL
                                  [-r [RUNTIME [RUNTIME ...]]] [-v] [--quiet]
                                  [--batch-size BATCH_SIZE]

Vortex exported IR graph validation pipeline; successful runs will produce
autogenerated reports

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to experiment config including dataset fields,
                        must be valid for validation, dataset.eval will be
                        used for evaluation
  -m MODEL, --model MODEL
                        path to IR model
  -r [RUNTIME [RUNTIME ...]], --runtime [RUNTIME [RUNTIME ...]]
                        runtime backend device
  -v, --verbose         verbose prediction output
  --quiet
  --batch-size BATCH_SIZE
                        batch size for validation; NOTE : passed value should
                        be matched with exported model batch size
```

E.g. :

```console
vortex ir_runtime_validate -c experiments/config/efficientnet_b0_classification_cifar10.yml \
                           -m experiments/outputs/efficientnet_b0_classification_cifar10/efficientnet_b0_classification_cifar10.pt \
                           -b 8 \
                           -r cpu cuda
```

This pipeline will generate several outputs :

- **Report file** : after successful evaluation, report file will be generated under directory `reports` in the **experiment directory** based on `experiment_name` under `output_directory`. Pro Tip : the generated report could be easily converted to pdf using [pandoc](https://pandoc.org/demos.html) or [vscode markdown-pdf extension](https://marketplace.visualstudio.com/items?itemName=yzane.markdown-pdf).

---

## IR Prediction Pipeline

This pipeline is used to test and visualize your IR model's (`*.pt` or `*.onnx`) prediction. If you need to integrate the prediction into your own script you can see the [IR prediction pipeline API section](../api/vortex.core.pipelines.md#irpredictionpipeline).

To run this pipeline, make sure you've already prepared :

- **IR model file** `*.pt` or `*.onnx` : obtained from [graph export pipeline](#graph-export-pipeline)
- **IR runtime library and environment** : make sure runtime library and environment is installed (currently runtime library installed together with vortex)
- **Input image(s)** : image file(s) (tested with `*.jpg`,`*.jpeg`,`*.png` extension)

You only need to run this command from the command line interface :

```console
usage: vortex ir_runtime_predict [-h] -m MODEL -i IMAGE [IMAGE ...]
                                 [-o OUTPUT_DIR]
                                 [--score_threshold SCORE_THRESHOLD]
                                 [--iou_threshold IOU_THRESHOLD] [-r RUNTIME]

Vortex IR model prediction pipeline; may receive multiple image(s) for batched
prediction

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to IR model
  -i IMAGE [IMAGE ...], --image IMAGE [IMAGE ...]
                        path to test image(s); at least 1 path should be
                        provided, supports up to model batch_size
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        directory to dump prediction visualization
  --score_threshold SCORE_THRESHOLD
                        score threshold for detection, only used if model is
                        detection, ignored otherwise
  --iou_threshold IOU_THRESHOLD
                        iou threshold for nms, only used if model is
                        detection, ignored otherwise
  -r RUNTIME, --runtime RUNTIME
                        runtime device

```

E.g. :

```console
vortex ir_runtime_predict -c experiments/config/efficientnet_b0_classification_cifar10.yml \
                          -m experiments/outputs/efficientnet_b0_classification_cifar10/efficientnet_b0_classification_cifar10.pt \
                          -i image1.jpg image2.jpg \
                          -r cuda \
                          -o output_vis
```

**NOTES** : Provided multiple input images will be treated as batch input. Vortex IR model is strict with batch size, means that provided input batch size must match with Vortex IR [`exporter`](../modules/exporter.md) batch size configuration.

This pipeline will generate several outputs :

- **Output Visualization Directory** : if `--output_dir` is provided, it will create the directory in your current working directory
- **Prediction Visualization** : prediction visualization will be generated in the `--output_dir` if provided, or in the current working dir if not. The generated file will have `prediction_` name prefix.