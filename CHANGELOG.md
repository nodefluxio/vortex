# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pytorch Lightning training pipeline
- registry class
- model base class
- add 'config.trainer.validation_interval' field with same functionality as 'config.validator.val_epoch'
- add 'config.trainer.logger' field with same functionality as 'config.logging'
- add 'config.dataloader.train' and 'config.dataloader.eval' field to support different args for train and val dataloader
- support for bool value for logger config, with default logger use tensorboard logger
- backbone base class
- backbone class for main models

### Changed
- `model_components` is removed, and changed with model base class
- deprecating 'config.seed', and move to 'config.trainer.seed'
- deprecating `stage` argument in `create_model`
- all defined backbones use backbone base class
- removed various old features affected by API redesign: `cli`, `predictor`, `pipelines`

### Fixed


## v0.2.1

### Added
- learning rate scheduler value plan visualizer script in [`scripts/visualize_learning_rate.py`](scripts/visualize_learning_rate.py)
- `save_best_metrics` config to save model checkpoint on best metrics
- always save checkpoint of last epoch model
- save `best_metrics` value in model checkpoint
- support for experimental [DETR model](https://github.com/facebookresearch/detr), this model is still unable to be exported caused by the limitation in current exporter design
- support for per parameter options (see [PyTorch Optimizer](https://pytorch.org/docs/stable/optim.html#per-parameter-options)) using `'param_groups'` key in model components.
- jit and export global context, see [`vortex/development/networks/modules/utils/config.py`](src/development/vortex/development/networks/modules/utils/config.py)
- support for per parameter options (see [PyTorch Optimizer](https://pytorch.org/docs/stable/optim.html#per-parameter-options)) 
using `'param_groups'` key in model components.
- support for None `additional_input` shape in model for scalar tensor input.
- support for image size type of list (w,h), or non-square image, in predict and export.
- validation loss calculation to experiment logger
- support for changing batch norm layer in all backbone with `norm_layer` argument.
- new backbone models: CSPNet, ResNest, RexNet, RegNet, TResNet. see more details in [spreadsheet](https://docs.google.com/spreadsheets/d/1M18Bm08P983_-5diHXAmlUmHusjpHdYMmcN0FmrNS74/edit?usp=sharing) or [csv summary](backbones_summary.txt)
- darknet weight converter in [scripts/convert_darknet_weight.py](scripts/convert_darknet_weight.py)
- yolov3 [converted darknet weight](experiments/outputs/yolov3_darknet_608) and [config file](experiments/configs/yolov3_darknet_608.yml)
- get model path from `config.checkpoint` if found and valid for `predict` and `export`
- yolov3 support for changing backbone stage to output with `backbone_stages` argument.

- `BasePredictionPipeline` argument to plot visualized result.
- new `predict` and `ir_runtime_predict` argument to not visualise and not save prediction result.
- new `list` command in main cli to list available components
- shape inference for onnx export

### Changed
- model checkpoint not save on hyperparameter optimization
- `save_epoch` config is not required
- Fix bug on scheduler.step() placement on trainer
- Update docs on supported Pytorch scheduler
- changed `StepLRWithBurnIn` to `StepLRWithWarmUp` scheduler
- changed `CosineLRScheduler` to `CosineLRWithWarmUp` scheduler
- changed `TanhLRScheduler` to `TanhLRWithWarmUp` scheduler
- `BaseTrainer.create_optimizer` only accept model parameters dict (e.g. from `model.parameters()`) instead of the model itself
- make `class_label` for detection not required
- update backbone components
- use mobilenetv3 definition and pretrained from [rwightman](https://github.com/rwightman/pytorch-image-models)
- improved cli help message
- improved progress bar, on `train` also display some metrics when training


### Fixed
- Fix error when using ir_runtime_validate with uneven batch splitting caused by different batch size on the last batch sample
- Fix error when `save_best_metrics` not present in experiment file
- Fix ONNX graph ops to convert `iou_threshold` as input
- lr scheduler bug when arguments changed and resumed
- yolov3 model definition


## 0.2.0

### Added

- Added new classification metrics for validation :
    - roc_curve
    - precision
    - recall
    - f1-score

- new checkpoint (saved) model format  
  checkpoint model is loaded as a dictionary with:  
  - default (required) member in checkpoint:
    - `epoch` --> number of epochs the model have been trained
    - `state_dict` -> trained model's state dict, this is the same as the entire data as the old format
    - `optimizer_state` --> optimizer state in training
    - `config` --> configuration used to train the model

  - additional (optional) member (if any):
    - `class_names` --> model's output class names map
    - `metrics` --> per epoch training metrics
    - `scheduler_state` -> `state_dict` for trainer's lr scheduler
- support for resume training in training pipeline using `checkpoint` field in config file and `--resume` flag in train command.
- `checkpoint` field in configuration file to point to the checkpoint model used to resume training if `--resume` flag is given. `init_state_dict` is still possible to be used but will automatically load `state_dict` to model even if `--resume` flag is not defined, but removed from docs.
- model checkpoint update script in `script/update_model.py`
- Now possible to access `class_names` and `input_specs` attributes from both `PytorchPredictionPipeline.model` and `IRPredictionPipeline.model`
- config deprecation checks for new config format
- Support for adding external model from user space
- Added `DALIDataLoader` and `nvidia_dali` augmentation module
  - `nvidia_dali` augmentation module must be used with `DALIDataLoader`
  - `DALIDataLoader` can use other augmentation module, however if `nvidia_dali` augmentation module if specified in the experiment file, it must be in the first order
  - `DALIDataLoader` utilize `ray` library to paralelize external module augmentation for each batch sample
- Added `DarknetDataset` to default dataset
- Added `dataset_visualizer.py` script


### Changed

- reading `class_names` attribute from checkpoint in prediction and export pipeline
- `class_names` in dataset, prediction, and export is optional. If not specified or `None`, will create a numbered class label `[class_0, class_1, ..., class_n]`
- `class_names` moved from `PytorchPredictionPipeline.class_names` to `PytorchPredictionPipeline.model.class_names` ( also applied to `IRPredictionPipeline`)
- None in string to normal python None
- new config format, as described in [#7](https://github.com/nodefluxio/vortex/issues/7).
  - change dataset name field `dataset` to `name` in `config.dataset.train` and `config.dataset.eval`.
  - move `device` from `config.trainer` to main `config`.
  - move `dataloader` field from `config.dataset` to main `config`.
  - change dataloader module from `dataloader` to `module` in `config.dataloader`.
  - change default dataloader module name from `DataLoader` to `PytorchDataLoader` in `config.dataloader.module`.
  - change `scheduler` to `lr_scheduler` in `config.trainer`.
  - change field `validation` to `validator` and move it from `config.trainer` to main `config`.
- Refactor `DatasetWrapper` into `BasicDatasetWrapper` and `DefaultDatasetWrapper`
- Change output type of BaseRuntime `__call__` method to list of orderedDict
- It is now possible to override image auto padding in dataset object by adding and set `self.disable_image_auto_pad = True` attribute on the `collate_fn` object provided by the `model_components`
- Refactor package into 2 namespace packages : `vortex.runtime` and `vortex.development`. In which case `vortex.development` is depend on `vortex.runtime`, but `vortex.runtime` can be installed as a standalone package for minimal requirement inferencing library


## v0.1.0

### Added

- Python package installation
- Command line interface
- Initial documentation
- Base development pipeline : training, validation, export, predict, ir-predict, ir-validation, hypopt
- Modular design
- Integration with image augmentation library ( Albumentations )
- Integration with hyperparameter optimization library ( Optuna )
- Integration with 3rd party experiment logger ( Comet.ml )
- Graph export to Torchscript and ONNX
- Visual report of model's performance and resource usage, see this example
- Various architecture support
  - 50 + infamous backbone networks
  - Classification and Detection architecture support
