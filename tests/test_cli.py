import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].joinpath('src', 'development')))

from vortex.development.version import __version__
from vortex.development.cli import parse_args
from vortex.development.command import (
    train, validate, predict,
    export, hypopt, 
    ir_runtime_predict,
    ir_runtime_validate
)

import pytest


def test_main_args(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--version"])
        out, _ = capsys.readouterr()
        assert out == __version__

def test_args_verbose():
    ## single verbose
    args = parse_args(["train", "config.yml", "-v"])
    assert args.verbose == 1
    args = parse_args(["train", "config.yml", "--verbose"])
    assert args.verbose == 1

    ## dual verbose
    args = parse_args(["train", "config.yml", "-vv"])
    assert args.verbose == 2
    args = parse_args(["train", "config.yml", "--verbose", "--verbose"])
    assert args.verbose == 2

    ## single quiet
    args = parse_args(["train", "config.yml", "-q"])
    assert args.quiet == 1
    args = parse_args(["train", "config.yml", "--quiet"])
    assert args.quiet == 1

    ## dual quiet
    args = parse_args(["train", "config.yml", "-qq"])
    assert args.quiet == 2
    args = parse_args(["train", "config.yml", "--quiet", "--quiet"])
    assert args.quiet == 2

    ## test error on verbose and quiet
    with pytest.raises(SystemExit):
        args = parse_args(["train", "config.yml", "-q", "-v"])


def test_train_args():
    cfg_name = "config.yml"

    ## normal run
    args = parse_args(["train", cfg_name, "--resume", "--no-log"])
    assert args.func == train.main
    assert args.cmd == "train"
    assert args.config == cfg_name
    assert args.resume
    assert args.no_log

    ## with deprecated args
    args = parse_args(["train", "--config", cfg_name, "--resume", "--no-log"])
    assert args.cmd == "train"
    assert args.config_dep == cfg_name and args.config == None
    assert args.resume
    assert args.no_log

    args = parse_args(["train", "-c", cfg_name, "--resume", "--no-log"])
    assert args.cmd == "train"
    assert args.config_dep == cfg_name and args.config == None
    assert args.resume
    assert args.no_log

    ## check deprecated
    with pytest.warns(DeprecationWarning):
        args = parse_args(["train", "--config", cfg_name])
        train.check_deprecated_args(args)
        assert args.config == cfg_name

    with pytest.warns(UserWarning):
        args = parse_args(["train", cfg_name, "--config", cfg_name])
        train.check_deprecated_args(args)
        assert args.config == cfg_name

    ## no config given
    with pytest.raises(RuntimeError):
        args = parse_args(["train"])
        train.check_deprecated_args(args)


def test_validate_args():
    cfg_name = "config.yml"
    weight_name = "ckpt.pth"

    ## normal run
    args = parse_args([
        "validate", cfg_name, "--weights", weight_name, "--batch-size", "4",
        "--devices", "cpu"
    ])
    assert args.func == validate.main
    assert args.cmd == "validate"
    assert args.config == cfg_name
    assert args.weights == weight_name
    assert args.batch_size == 4
    assert args.devices == ["cpu"]

    args = parse_args([
        "validate", cfg_name, "-w", weight_name, "-b", "4", "-d", "cpu"
    ])
    assert args.func == validate.main
    assert args.cmd == "validate"
    assert args.config == cfg_name
    assert args.weights == weight_name
    assert args.batch_size == 4
    assert args.devices == ["cpu"]

    ## with deprecated args
    args = parse_args([
        "validate", "--config", cfg_name, "--weights", weight_name, "--batch-size", "4",
        "--devices", "cpu"
    ])
    assert args.cmd == "validate"
    assert args.config_dep == cfg_name and args.config == None
    assert args.weights == weight_name
    assert args.batch_size == 4
    assert args.devices == ["cpu"]

    args = parse_args(["validate", "-c", cfg_name])
    assert args.cmd == "validate"
    assert args.config_dep == cfg_name and args.config == None

    ## devices
    args = parse_args([
        "validate", cfg_name, "--devices", "cpu", "cpu"
    ])
    assert args.func == validate.main
    assert args.cmd == "validate"
    assert args.config == cfg_name
    assert args.devices == ["cpu", "cpu"]

    ## check deprecated
    with pytest.warns(DeprecationWarning):
        args = parse_args(["validate", "--config", cfg_name])
        validate.check_deprecated_args(args)
        assert args.config == cfg_name

    with pytest.warns(UserWarning):
        args = parse_args(["validate", cfg_name, "--config", cfg_name])
        validate.check_deprecated_args(args)
        assert args.config == cfg_name

    ## no config given
    with pytest.raises(RuntimeError):
        args = parse_args(["validate"])
        validate.check_deprecated_args(args)


def test_predict_args():
    cfg_name = "config.yml"
    img_name = "img.jpg"
    weight_name = "ckpt.pth"
    out_dir = "tests"

    ## normal run
    args = parse_args([
        "predict", cfg_name, img_name, "--weights", weight_name,
        "--output-dir", out_dir, "--device", "cpu", 
        "--no-visualize", "--no-save",
        "--score_threshold", "0.5", "--iou_threshold", "0.45"
    ])
    assert args.func == predict.main
    assert args.cmd == "predict"
    assert args.config == cfg_name
    assert args.image == [img_name]
    assert args.weights == weight_name
    assert args.output_dir == out_dir
    assert args.device == "cpu"
    assert args.no_visualize and args.no_save
    assert args.score_threshold == 0.5
    assert args.iou_threshold == 0.45

    args = parse_args([
        "predict", cfg_name, img_name, "-w", weight_name,
        "-o", out_dir, "-d", "cpu"
    ])
    assert args.func == predict.main
    assert args.cmd == "predict"
    assert args.config == cfg_name
    assert args.image == [img_name]
    assert args.weights == weight_name
    assert args.output_dir == out_dir
    assert args.device == "cpu"
    assert not args.no_visualize and not args.no_save
    assert args.score_threshold == 0.9
    assert args.iou_threshold == 0.2

    ## with deprecated args
    args = parse_args(["predict", "--config", cfg_name, "--image", img_name])
    assert args.cmd == "predict"
    assert args.config_dep == cfg_name and args.config == None
    assert args.image_dep == [img_name] and args.image == []

    args = parse_args(["predict", "-c", cfg_name, "-i", img_name])
    assert args.cmd == "predict"
    assert args.config_dep == cfg_name and args.config == None
    assert args.image_dep == [img_name] and args.image == []

    ## check deprecated
    with pytest.warns(DeprecationWarning) as record:
        args = parse_args(["predict", "--config", cfg_name, "--image", img_name])
        predict.check_deprecated_args(args)
        assert args.config == cfg_name
        assert args.image == [img_name]
        assert len(record) == 2

    with pytest.warns(UserWarning) as record:
        args = parse_args([
            "predict", cfg_name, img_name, "--config", cfg_name,
            "--image", img_name
        ])
        predict.check_deprecated_args(args)
        assert args.config == cfg_name
        assert args.image == [img_name]
        assert len(record) == 2

    ## no config given
    with pytest.raises(RuntimeError):
        args = parse_args(["predict"])
        predict.check_deprecated_args(args)

    ## only config given
    with pytest.raises(RuntimeError):
        args = parse_args(["predict", cfg_name])
        predict.check_deprecated_args(args)


def test_export_args():
    cfg_name = "config.yml"
    weight_name = "ckpt.pth"
    input_name = "image.jpg"

    ## normal run
    args = parse_args([
        "export", cfg_name, "--weights", weight_name,
        "--example-input", input_name
    ])
    assert args.func == export.main
    assert args.cmd == "export"
    assert args.config == cfg_name
    assert args.weights == weight_name
    assert args.example_input == input_name

    args = parse_args([
        "export", cfg_name, "-w", weight_name, "-i", input_name
    ])
    assert args.func == export.main
    assert args.cmd == "export"
    assert args.config == cfg_name
    assert args.weights == weight_name
    assert args.example_input == input_name

    ## with deprecated args
    args = parse_args(["export", "--config", cfg_name])
    assert args.cmd == "export"
    assert args.config_dep == cfg_name and args.config == None

    args = parse_args(["export", "-c", cfg_name])
    assert args.cmd == "export"
    assert args.config_dep == cfg_name and args.config == None

    ## check deprecated
    with pytest.warns(DeprecationWarning):
        args = parse_args(["export", "--config", cfg_name])
        export.check_deprecated_args(args)
        assert args.config == cfg_name

    with pytest.warns(UserWarning):
        args = parse_args(["export", cfg_name, "--config", cfg_name])
        export.check_deprecated_args(args)
        assert args.config == cfg_name

    ## no config given
    with pytest.raises(RuntimeError):
        args = parse_args(["export"])
        export.check_deprecated_args(args)


def test_hypopt_args():
    cfg_name = "config.yml"
    optcfg_name = "optconfig.yml"
    weight_name = "ckpt.pth"

    ## normal run
    args = parse_args([
        "hypopt", cfg_name, optcfg_name, "--weights", weight_name
    ])
    assert args.func == hypopt.main
    assert args.cmd == "hypopt"
    assert args.config == cfg_name
    assert args.optconfig == optcfg_name
    assert args.weights == weight_name

    args = parse_args([
        "hypopt", cfg_name, optcfg_name, "-w", weight_name
    ])
    assert args.func == hypopt.main
    assert args.cmd == "hypopt"
    assert args.config == cfg_name
    assert args.weights == weight_name

    ## with deprecated args
    args = parse_args([
        "hypopt", "-c", cfg_name, "-o", optcfg_name
    ])
    assert args.cmd == "hypopt"
    assert args.config_dep == cfg_name and args.config == None
    assert args.optconfig_dep == optcfg_name and args.optconfig == None

    ## check deprecated
    with pytest.warns(DeprecationWarning) as record:
        args = parse_args([
            "hypopt", "--config", cfg_name, "--optconfig", optcfg_name
        ])
        assert args.config_dep == cfg_name and args.config == None
        assert args.optconfig_dep == optcfg_name and args.optconfig == None

        hypopt.check_deprecated_args(args)
        assert args.config == cfg_name
        assert args.optconfig == optcfg_name
        assert len(record) == 2

    with pytest.warns(UserWarning) as record:
        args = parse_args([
            "hypopt", cfg_name, optcfg_name, "--config", cfg_name,
            "--optconfig", optcfg_name
        ])
        hypopt.check_deprecated_args(args)
        assert args.config == cfg_name
        assert args.optconfig == optcfg_name
        assert len(record) == 2

    ## no config given
    with pytest.raises(RuntimeError):
        args = parse_args(["hypopt"])
        hypopt.check_deprecated_args(args)

    ## only config given
    with pytest.raises(RuntimeError):
        args = parse_args(["hypopt", cfg_name])
        hypopt.check_deprecated_args(args)


def test_ir_validate_args():
    cfg_name = "config.yml"
    model_name = "ckpt.pth"

    ## normal run
    args = parse_args([
        "ir_runtime_validate", cfg_name, model_name, "--batch-size", "4",
        "--runtime", "cpu"
    ])
    assert args.func == ir_runtime_validate.main
    assert args.cmd == "ir_runtime_validate"
    assert args.config == cfg_name
    assert args.model == model_name
    assert args.batch_size == 4
    assert args.runtime == ["cpu"]

    args = parse_args([
        "ir_runtime_validate", cfg_name, model_name, "-b", "4", "-r", "cpu"
    ])
    assert args.func == ir_runtime_validate.main
    assert args.cmd == "ir_runtime_validate"
    assert args.config == cfg_name
    assert args.model == model_name
    assert args.batch_size == 4
    assert args.runtime == ["cpu"]

    ## with deprecated args
    args = parse_args(["ir_runtime_validate", "-c", cfg_name, "-m", model_name])
    assert args.cmd == "ir_runtime_validate"
    assert args.config_dep == cfg_name and args.config == None
    assert args.model_dep == model_name and args.model == None

    ## runtime
    args = parse_args([
        "ir_runtime_validate", cfg_name, model_name, "--runtime", "cpu", "cpu"
    ])
    assert args.func == ir_runtime_validate.main
    assert args.cmd == "ir_runtime_validate"
    assert args.config == cfg_name
    assert args.runtime == ["cpu", "cpu"]

    ## check deprecated
    with pytest.warns(DeprecationWarning) as record:
        args = parse_args([
            "ir_runtime_validate", "--config", cfg_name, "--model", model_name
        ])
        assert args.config_dep == cfg_name and args.config == None
        assert args.model_dep == model_name and args.model == None

        ir_runtime_validate.check_deprecated_args(args)
        assert args.config == cfg_name
        assert args.model == model_name
        assert len(record) == 2

    with pytest.warns(UserWarning) as record:
        args = parse_args([
            "ir_runtime_validate", cfg_name, model_name, "--config", cfg_name,
            "--model", model_name
        ])
        ir_runtime_validate.check_deprecated_args(args)
        assert args.config == cfg_name
        assert args.model == model_name
        assert len(record) == 2

    ## no config given
    with pytest.raises(RuntimeError):
        args = parse_args(["ir_runtime_validate"])
        ir_runtime_validate.check_deprecated_args(args)

    ## only config given
    with pytest.raises(RuntimeError):
        args = parse_args(["ir_runtime_validate", cfg_name])
        ir_runtime_validate.check_deprecated_args(args)


def test_ir_predict_args():
    model_name = "model.onnx"
    img_name = "img.jpg"
    out_dir = "tests"

    ## normal run
    args = parse_args([
        "ir_runtime_predict", model_name, img_name, "--output-dir", out_dir, 
        "--runtime", "cpu", "--no-visualize", "--no-save",
        "--score_threshold", "0.5", "--iou_threshold", "0.45"
    ])
    assert args.func == ir_runtime_predict.main
    assert args.cmd == "ir_runtime_predict"
    assert args.model == model_name
    assert args.image == [img_name]
    assert args.output_dir == out_dir
    assert args.runtime == "cpu"
    assert args.no_visualize and args.no_save
    assert args.score_threshold == 0.5
    assert args.iou_threshold == 0.45

    args = parse_args([
        "ir_runtime_predict", model_name, img_name, "-o", out_dir, "-r", "cpu"
    ])
    assert args.func == ir_runtime_predict.main
    assert args.cmd == "ir_runtime_predict"
    assert args.model == model_name
    assert args.image == [img_name]
    assert args.output_dir == out_dir
    assert args.runtime == "cpu"
    assert not args.no_visualize and not args.no_save
    assert args.score_threshold == 0.9
    assert args.iou_threshold == 0.2

    ## with deprecated args
    args = parse_args(["ir_runtime_predict", "-m", model_name, "-i", img_name])
    assert args.cmd == "ir_runtime_predict"
    assert args.model_dep == model_name and args.model == None
    assert args.image_dep == [img_name] and args.image == []

    ## check deprecated
    with pytest.warns(DeprecationWarning) as record:
        args = parse_args(["ir_runtime_predict", "--model", model_name, "--image", img_name])
        assert args.cmd == "ir_runtime_predict"
        assert args.model_dep == model_name and args.model == None
        assert args.image_dep == [img_name] and args.image == []

        ir_runtime_predict.check_deprecated_args(args)
        assert args.model == model_name
        assert args.image == [img_name]
        assert len(record) == 2

    with pytest.warns(UserWarning) as record:
        args = parse_args([
            "ir_runtime_predict", model_name, img_name, "--model", model_name,
            "--image", img_name
        ])
        ir_runtime_predict.check_deprecated_args(args)
        assert args.model == model_name
        assert args.image == [img_name]
        assert len(record) == 2

    ## no model given
    with pytest.raises(RuntimeError):
        args = parse_args(["ir_runtime_predict"])
        ir_runtime_predict.check_deprecated_args(args)

    ## only model given
    with pytest.raises(RuntimeError):
        args = parse_args(["ir_runtime_predict", model_name])
        ir_runtime_predict.check_deprecated_args(args)
