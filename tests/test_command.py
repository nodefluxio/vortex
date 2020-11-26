import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].joinpath('src', 'development')))

from vortex.development import cli
from vortex.development.command import list as list_cmd

import pytest
import shutil
import fnmatch

test_dir = Path(__file__).parent
CONFIG_PATH = test_dir.joinpath("config", "test_classification_pipelines.yml")
OPTCONFIG_PATH = test_dir.joinpath("config", "test_hypopt_train_objective.yml")
IMG_SAMPLE = test_dir.joinpath("test_dataset", "classification", "val", "cat", "1.jpeg")
OUTPUT_DIR = test_dir.joinpath("output_test")
EXP_DIR = OUTPUT_DIR.joinpath("test_classification_pipelines")


@pytest.fixture(scope="module")
def training_result():  ## only do training once with fixture
    if EXP_DIR.exists():
        shutil.rmtree(EXP_DIR)
        EXP_DIR.mkdir(parents=True)

    cli.main(["train", str(CONFIG_PATH)])
    exp_dir_content = [d for d in EXP_DIR.iterdir() if d.is_dir()]
    yield exp_dir_content[0]
    shutil.rmtree(EXP_DIR)

@pytest.fixture(scope="module")
def exported_models(training_result):
    cli.main(["export", str(CONFIG_PATH)])
    exp_dir_content = [
        f for f in EXP_DIR.iterdir() if f.is_file() and f.suffix in ['.onnx', '.pt']
    ]
    return exp_dir_content


def test_train(training_result):
    assert training_result.exists()
    assert training_result.joinpath("config.yml").exists()

    ## fail to resume, need to specify checkpoint path
    with pytest.raises(RuntimeError):
        cli.main(["train", str(CONFIG_PATH), "--resume"])


def test_validate(training_result):
    assert EXP_DIR.joinpath(EXP_DIR.name + '.pth').exists()

    cli.main(["validate", str(CONFIG_PATH)])
    assert EXP_DIR.joinpath("reports").exists()
    assert EXP_DIR.joinpath("reports", EXP_DIR.name + '_validation_cpu.md').exists()

    weight_path = training_result.joinpath(EXP_DIR.name + '.pth')
    cli.main(["validate", str(CONFIG_PATH), "--weights", str(weight_path)])


def test_predict(training_result):
    assert EXP_DIR.joinpath(EXP_DIR.name + '.pth').exists()

    ## single image
    cli.main([
        "predict", str(CONFIG_PATH), str(IMG_SAMPLE), "--no-visualize",
        "--output-dir", str(EXP_DIR)
    ])
    assert EXP_DIR.joinpath("prediction_1.jpeg")

    cli.main([
        "predict", str(CONFIG_PATH), str(IMG_SAMPLE), str(IMG_SAMPLE), 
        "--no-visualize", "--output-dir", str(EXP_DIR)
    ])

    weight_path = training_result.joinpath(EXP_DIR.name + '.pth')
    cli.main([
        "predict", str(CONFIG_PATH), str(IMG_SAMPLE), "--weights", str(weight_path),
        "--no-visualize", "--output-dir", str(EXP_DIR)
    ])

def test_hypopt():
    cli.main(["hypopt", str(CONFIG_PATH), str(OPTCONFIG_PATH)])
    hypopt_out_dir = EXP_DIR.joinpath("hypopt", "optimizer_search")
    assert hypopt_out_dir.exists()
    assert hypopt_out_dir.joinpath("best_params.txt").exists()


def test_export(training_result, exported_models):
    assert EXP_DIR.joinpath(EXP_DIR.name + '.pth').exists()

    for model_path in exported_models:
        assert model_path.exists()

    weight_path = training_result.joinpath(EXP_DIR.name + '.pth')
    cli.main(["export", str(CONFIG_PATH), "--weights", str(weight_path)])


def test_ir_validate(exported_models):
    assert EXP_DIR.joinpath(EXP_DIR.name + '.pth').exists()

    onnx_models = [
        m for m in exported_models if m.suffix == '.onnx' and not m.stem.endswith('_bs8')
    ]
    torchscript_models = [
        m for m in exported_models if m.suffix == '.pt' and not m.stem.endswith('_bs8')
    ]

    cli.main(["ir_runtime_validate", str(CONFIG_PATH), str(onnx_models[0])])
    assert EXP_DIR.joinpath("reports").exists()
    reports_name = EXP_DIR.name + '_onnx_IR_validation_cpu.md'
    assert EXP_DIR.joinpath("reports", reports_name).exists()

    cli.main(["ir_runtime_validate", str(CONFIG_PATH), str(torchscript_models[0])])
    assert EXP_DIR.joinpath("reports").exists()
    reports_name = EXP_DIR.name + '_torchscript_IR_validation_cpu.md'
    assert EXP_DIR.joinpath("reports", reports_name).exists()


def test_ir_predict(exported_models):
    assert EXP_DIR.joinpath(EXP_DIR.name + '.pth').exists()

    onnx_models = [
        m for m in exported_models if m.suffix == '.onnx' and not m.stem.endswith('_bs8')
    ]
    torchscript_models = [
        m for m in exported_models if m.suffix == '.pt' and not m.stem.endswith('_bs8')
    ]

    ## single image
    cli.main([
        "ir_runtime_predict", str(onnx_models[0]), str(IMG_SAMPLE), "--no-visualize",
        "--output-dir", str(EXP_DIR)
    ])
    assert EXP_DIR.joinpath("onnx_ir_prediction_1.jpeg")

    ## error on multiple image, as model only single batch
    with pytest.raises(AssertionError):
        cli.main([
            "ir_runtime_predict", str(onnx_models[0]), str(IMG_SAMPLE), str(IMG_SAMPLE), 
            "--no-visualize", "--output-dir", str(EXP_DIR)
        ])

    cli.main([
        "ir_runtime_predict", str(torchscript_models[0]), str(IMG_SAMPLE),
        "--no-visualize", "--output-dir", str(EXP_DIR)
    ])
    assert EXP_DIR.joinpath("torchscript_ir_prediction_1.jpeg")


def test_list_backbone(capsys):
    list_backbone = list_cmd.BackboneList()

    ## get family
    to_get = ['resnet', 'darknet']
    result = list_backbone._get_family(to_get)
    assert sorted(result.keys()) == sorted(to_get)

    with pytest.raises(AssertionError):
        list_backbone._get_family('resnet')

    ## one not available
    to_get = ['resnet', 'mobilenet'] # mobilenet is not available
    result = list_backbone._get_family(to_get)
    assert list(result.keys()) == ['resnet']
    printed, _ = capsys.readouterr()
    assert "'mobilenet' family is not available." in printed

    ## two not available
    to_get = ['resnet', 'mobilenet', "darkent"]
    result = list_backbone._get_family(to_get)
    assert list(result.keys()) == ['resnet']
    printed, _ = capsys.readouterr()
    assert "'mobilenet', and 'darkent' family is not available." in printed

    ## three not available
    to_get = ['renset', 'mobilenet', "darkent"]
    result = list_backbone._get_family(to_get)
    assert result == {}
    printed, _ = capsys.readouterr()
    assert "'renset', 'mobilenet', and 'darkent' family is not available." in printed

    ## filter
    data = list_backbone.available_backbone.copy()
    pattern = '*resne*t*'
    filtered = list_backbone._filter(data, pattern)
    expected = {
        n: [v for v in val if fnmatch.fnmatchcase(v, pattern)] for n, val in data.items()
    }
    expected = {n: v for n,v in expected.items() if v}
    assert filtered == expected

    ##  cli
    cli.main([
        "list", "backbone", "--family", "resnet", "resnest",
        "--filter", "resne*t*"
    ])

def test_list_model_dataset():
    ## just make sure it is running properly for now
    ## TODO: do more testing

    cli.main(["list", "model"])
    cli.main(["list", "dataset"])
