# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- support for resume training in training pipeline
- model checkpoint update script in `script/update_model.py`

### Changed

- reading `class_names` attribute from checkpoint in prediction and export pipeline
- `class_names` in dataset, prediction, and export is optional. If not specified or `None`, will create a numbered class label `[class_0, class_1, ..., class_n]`



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
