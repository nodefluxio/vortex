# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
