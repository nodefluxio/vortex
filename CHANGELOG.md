# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- learning rate scheduler value plan visualizer script in [`scripts/visualize_learning_rate.py`](scripts/visualize_learning_rate.py)

### Changed

### Fixed
- Fix error when using ir_runtime_validate with uneven batch splitting caused by different batch size on the last batch sample

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
