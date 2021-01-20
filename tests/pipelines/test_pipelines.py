import os
import shutil
import pytest
import numpy as np
import cv2
import yaml
import torch

from pathlib import Path
from easydict import EasyDict
from collections import OrderedDict
from copy import deepcopy

from vortex.development.pipelines import (
    TrainingPipeline,
    PytorchValidationPipeline,
    PytorchPredictionPipeline,
    IRValidationPipeline,
    GraphExportPipeline,
    IRPredictionPipeline,
    HypOptPipeline
)
from vortex.development.utils.factory import create_model
from vortex.development.utils.parser.parser import load_config
from vortex.development.utils.parser.loader import Loader

from ..common import state_dict_is_equal


config_path = "tests/config/test_classification_pipelines.yml"
config_old_path = "tests/config/test_classification_pipelines_old.yml"
hypopt_train_obj_path = "tests/config/test_hypopt_train_objective.yml"
onnx_model_path = "tests/output_test/test_classification_pipelines/test_classification_pipelines.onnx"
pt_model_path = "tests/output_test/test_classification_pipelines/test_classification_pipelines.pt"
pth_model_path = "tests/output_test/test_classification_pipelines/test_classification_pipelines.pth"

# Load configuration from experiment file
config = load_config(config_path)
with open(hypopt_train_obj_path) as f:
    hypopt_train_obj_config = EasyDict(yaml.load(f, Loader=Loader))


class InfoPlaceHolder():
    def __init__(self):
        self.run_directory = '.'

train_info = InfoPlaceHolder()
train_info_sch = InfoPlaceHolder()


class TestTrainingPipeline():

    cfg_scheduler = deepcopy(config)
    cfg_scheduler.trainer.lr_scheduler = {
        'method': 'StepLR',
        'args': {'step_size': 1}
    }


    @pytest.mark.parametrize(
        "device", [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU"),
            id="gpu"
        )
    ])
    def test_train_device(self, device):
        orig_cfg_dev = deepcopy(config)
        orig_cfg_dev.device = device
        old_cfg_dev = deepcopy(config)
        old_cfg_dev.trainer.device = device
        old_cfg_dev.pop('device', None)

        for cfg_dev in (old_cfg_dev, orig_cfg_dev):
            train_executor = TrainingPipeline(config=cfg_dev, config_path=config_path, hypopt=False)
            assert train_executor.device == device
            model_device = list(train_executor.model_components.network.parameters())[0].device
            assert model_device == torch.device(device)
            loss_param = list(train_executor.criterion.parameters())
            if len(loss_param):
                loss_device = loss_param[0].device
                assert loss_device == torch.device(device)


    @pytest.mark.parametrize("scheduler", [
        pytest.param(False, id="no scheduler"),
        pytest.param(True, id="with scheduler")
    ])
    def test_fresh_train(self, scheduler):
        if not scheduler: # Clear existing output file on first run only
            if Path(config.output_directory).exists():
                shutil.rmtree(Path(config.output_directory))
            if Path('experiments/local_runs.log').exists():
                os.remove(Path('experiments/local_runs.log'))

        # Instantiate Training
        if scheduler:
            cfg = self.cfg_scheduler
            info_holder = train_info_sch
        else:
            cfg = config
            info_holder = train_info

        train_executor = TrainingPipeline(config=cfg, config_path=config_path, hypopt=False)
        info_holder.run_directory = train_executor.run_directory

        output = train_executor.run()

        # Check output type
        assert isinstance(output, EasyDict)

        # Check every enforce key
        assert 'epoch_losses' in output.keys()
        assert 'val_metrics' in output.keys()
        assert 'learning_rates' in output.keys()

        epoch_losses = output.epoch_losses
        val_metrics = output.val_metrics
        learning_rates = output.learning_rates

        # Make sure returned values is list based on each epoch process
        assert isinstance(epoch_losses,list)
        assert len(epoch_losses) == config.trainer.epoch
        assert isinstance(val_metrics,list)
        assert len(val_metrics) == int(config.trainer.epoch/config.validator.val_epoch)
        assert isinstance(learning_rates,list)
        assert len(learning_rates) == config.trainer.epoch

        # Make sure every desired output exist
        run_dir = Path(train_executor.run_directory)
        experiment_dir = Path(train_executor.experiment_directory)

        # Check experiment directory and run directory exist
        assert run_dir.exists()

        # Check if config is duplicated as backup in run directory
        assert run_dir.joinpath('config.yml').exists()

        # Check saved weight
        final_weight = experiment_dir.joinpath('{}.pth'.format(config.experiment_name))
        epoch_weight = run_dir.joinpath("{}-epoch-0.pth".format(config.experiment_name))
        last_weight = run_dir.joinpath("{}.pth".format(config.experiment_name))
        best_loss_weight = run_dir.joinpath("{}-best-loss.pth".format(config.experiment_name))
        best_acc_weight = run_dir.joinpath("{}-best-accuracy.pth".format(config.experiment_name))
        best_prec_weight = run_dir.joinpath("{}-best-precision (micro).pth".format(config.experiment_name))
        assert final_weight.exists() and epoch_weight.exists() and last_weight.exists()
        assert best_loss_weight.exists() and best_acc_weight.exists() and best_prec_weight.exists()

        # Check local_runs log is generated
        assert Path('experiments/local_runs.log').exists()

        ## check saved model checkpoint
        ckpt = torch.load(final_weight)
        required_ckpt = ('epoch', 'state_dict', 'optimizer_state', 'class_names', 'config')
        assert all((k in ckpt) for k in required_ckpt)
        assert ckpt['config'] == cfg
        assert ckpt['epoch'] == cfg.trainer.epoch
        assert tuple(ckpt['class_names']) == ('cat', 'dog')
        assert not 'scheduler_state' in ckpt if not scheduler else 'scheduler_state' in ckpt
        assert 'best_metrics' in ckpt

        ckpt = torch.load(best_loss_weight)
        assert ckpt['best_metrics']['loss'] == train_executor.best_metrics['loss']
        ckpt = torch.load(best_acc_weight)
        assert ckpt['best_metrics']['accuracy'] == train_executor.best_metrics['accuracy']

    def test_fresh_train_with_ckpt(self):
        # Instantiate Training
        cfg = deepcopy(config)
        cfg.device = "cpu"
        cfg.checkpoint = train_info.run_directory/'test_classification_pipelines-epoch-0.pth'
        train_executor = TrainingPipeline(config=cfg, config_path=config_path, hypopt=False)

        ckpt = torch.load(cfg.checkpoint)
        assert state_dict_is_equal(ckpt['state_dict'], train_executor.model_components.network.state_dict())

        output = train_executor.run()

        # Check outputs
        assert isinstance(output, EasyDict)
        assert all(x in output for x in ('epoch_losses', 'val_metrics', 'learning_rates'))

        epoch_losses = output.epoch_losses
        val_metrics = output.val_metrics
        learning_rates = output.learning_rates

        # Make sure returned values is list based on each epoch process
        assert isinstance(epoch_losses,list)
        assert len(epoch_losses) == config.trainer.epoch
        assert isinstance(val_metrics,list)
        assert len(val_metrics) == int(config.trainer.epoch/config.validator.val_epoch)
        assert isinstance(learning_rates,list)
        assert len(learning_rates) == config.trainer.epoch

        # Check experiment directory and run directory exist
        train_executor.run_directory = Path(train_executor.run_directory)
        assert train_executor.run_directory.exists()

        # Check if config is duplicated as backup in run directory
        assert train_executor.run_directory.joinpath("config.yml").exists()

        # Check if final weight is generated when training ends
        final_weight = Path(train_executor.experiment_directory) / '{}.pth'.format(config.experiment_name)
        assert Path(final_weight).exists()

        # Check local_runs log is generated
        assert Path('experiments/local_runs.log').exists()

        ## check saved model checkpoint
        ckpt = torch.load(final_weight)
        required_ckpt = ('epoch', 'state_dict', 'optimizer_state', 'class_names', 'config')
        assert all((k in ckpt) for k in required_ckpt)
        assert ckpt['config'] == cfg
        assert tuple(ckpt['class_names']) == ('cat', 'dog')
        assert not 'scheduler_state' in ckpt

    @pytest.mark.parametrize("scheduler", [
        pytest.param(False, id="no scheduler"),
        pytest.param(True, id="with scheduler")
    ])
    def test_continue_train(self, scheduler):
        if scheduler:
            cfg = self.cfg_scheduler
            info_holder = train_info_sch
        else:
            cfg = deepcopy(config)
            info_holder = train_info
        cfg.device = "cpu"
        cfg.checkpoint = info_holder.run_directory/'test_classification_pipelines-epoch-0.pth'
        train_executor = TrainingPipeline(config=cfg, config_path=config_path, hypopt=False, resume=True)

        ckpt = torch.load(cfg.checkpoint)
        assert state_dict_is_equal(ckpt['state_dict'], train_executor.model_components.network.state_dict())
        assert ckpt['best_metrics'] == train_executor.best_metrics

        output = train_executor.run()

        # Check output type
        assert isinstance(output, EasyDict)

        # Check every enforce key
        assert 'epoch_losses' in output.keys()
        assert 'val_metrics' in output.keys()
        assert 'learning_rates' in output.keys()

        epoch_losses = output.epoch_losses
        val_metrics = output.val_metrics
        learning_rates = output.learning_rates

        # Make sure returned values is list based on each epoch process
        rest_of_epoch = config.trainer.epoch - train_executor.start_epoch
        assert isinstance(epoch_losses,list)
        assert len(epoch_losses) == rest_of_epoch
        assert isinstance(val_metrics,list)
        assert len(val_metrics) == int(rest_of_epoch/config.validator.val_epoch)
        assert isinstance(learning_rates,list)
        assert len(learning_rates) == rest_of_epoch

        # Make sure every desired output exist

        # Check experiment directory and run directory exist
        assert Path(train_executor.run_directory).exists()

        # Check if config is duplicated as backup in run directory
        backup_config = Path(train_executor.run_directory)/'config.yml'
        assert Path(backup_config).exists()

        # Check if final weight is generated when training ends
        final_weight = Path(train_executor.experiment_directory) / '{}.pth'.format(config.experiment_name)
        assert Path(final_weight).exists()

        # Check local_runs log is generated
        assert Path('experiments/local_runs.log').exists()

        ## check saved model checkpoint
        ckpt = torch.load(final_weight)
        required_ckpt = ('epoch', 'state_dict', 'optimizer_state', 'class_names', 'config')
        assert all((k in ckpt) for k in required_ckpt)
        assert ckpt['config'] == cfg
        assert tuple(ckpt['class_names']) == ('cat', 'dog')

        assert state_dict_is_equal(ckpt['state_dict'], train_executor.model_components.network.state_dict())
        assert state_dict_is_equal(ckpt['optimizer_state'], train_executor.trainer.optimizer.state_dict())
        if train_executor.trainer.scheduler is not None:
            assert 'scheduler_state' in ckpt
            assert state_dict_is_equal(ckpt['scheduler_state'], train_executor.trainer.scheduler.state_dict())
        else:
            assert not 'scheduler_state' in ckpt


def test_create_model():
    def _check(model):
        assert all(x in model for x in ('network', 'preprocess', 'postprocess'))
        assert all(x in model for x in ('loss', 'collate_fn'))

    model = create_model(config.model, stage='train')
    _check(model)

    ckpt = torch.load(pth_model_path)

    ## using 'state_dict' path as 'str'
    model = create_model(config.model, state_dict=pth_model_path, stage='train')
    _check(model)
    state_dict_is_equal(ckpt['state_dict'], model.network.state_dict())

    ## using 'state_dict' path as 'Path'
    model = create_model(config.model, state_dict=Path(pth_model_path), stage='train')
    _check(model)
    state_dict_is_equal(ckpt['state_dict'], model.network.state_dict())

    ## using 'state_dict' from ckpt
    model = create_model(config.model, state_dict=ckpt['state_dict'], stage='train')
    _check(model)
    state_dict_is_equal(ckpt['state_dict'], model.network.state_dict())

    ## using 'init_state_dict' from config
    new_cfg_model = deepcopy(config.model)
    new_cfg_model['init_state_dict'] = pth_model_path
    model = create_model(new_cfg_model, stage='train')
    _check(model)
    state_dict_is_equal(ckpt['state_dict'], model.network.state_dict())

    ## if both 'init_state_dict' config and 'state_dict' argument is specified
    ## 'state_dict' argument should be of top priority
    ckpt_ep0 = torch.load(train_info.run_directory/'test_classification_pipelines-epoch-0.pth')
    model = create_model(new_cfg_model, state_dict=ckpt_ep0['state_dict'], stage='train')
    _check(model)
    state_dict_is_equal(ckpt_ep0['state_dict'], model.network.state_dict())


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU"),
            id="gpu"
        )
    ]
)
def test_validation_pipeline_device(device):
    orig_cfg_dev = deepcopy(config)
    orig_cfg_dev.device = device
    old_cfg_dev = deepcopy(config)
    old_cfg_dev.trainer.device = device
    old_cfg_dev.pop('device', None)

    for cfg_dev in (old_cfg_dev, orig_cfg_dev):
        predictor = PytorchValidationPipeline(config=cfg_dev, weights=None, backends=[device])
        assert torch.device(predictor.backends[0]) == torch.device(device)


def test_validation_pipeline():
    val_cfg = deepcopy(config)

    validation_executor = PytorchValidationPipeline(
        config=val_cfg, weights = None, 
        backends = 'cpu',
        generate_report = True
    )

    eval_results = validation_executor.run(batch_size=2)

    # Check return value
    assert isinstance(eval_results,EasyDict)

    # Check generated reports
    report_dir = Path(config.output_directory) / config.experiment_name / 'reports'
    generated_report = Path(report_dir) / '{}_validation_cpu.md'.format(config.experiment_name)
    assert generated_report.exists()


class TestPredictionPipeline():

    # Clear pre-existing output file
    if Path('tests/output_predict_test').exists():
        shutil.rmtree(Path('tests/output_predict_test'))
    
    weight_file = "tests/output_test/test_classification_pipelines/test_classification_pipelines.pth"

    def _check_result(self, results, visualize=True):
        # Prediction pipeline output must be EasyDict
        assert isinstance(results, EasyDict)
        assert 'prediction' in results
        assert 'visualization' in results

        ## Prediction output must be in a list
        ## representation of batched output list index [0] means result for input [0]
        assert isinstance(results.prediction, list)
        assert isinstance(results.prediction[0], EasyDict)
        if visualize:
            assert isinstance(results.visualization, list)
            assert isinstance(results.visualization[0], np.ndarray)
        else:
            assert results.visualization is None

    def _check_pipeline(self, pipeline):
        ckpt = torch.load(pth_model_path)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        assert state_dict_is_equal(state_dict, pipeline.model.model.state_dict())

    def test_model_api(self):
        vortex_predictor = PytorchPredictionPipeline(config = config,
                                        weights = None,
                                        device = 'cpu')

        assert isinstance(vortex_predictor.model.input_specs, OrderedDict)
        assert isinstance(vortex_predictor.model.class_names, list)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
            )
        ]
    )
    def test_device(self, device):
        orig_cfg_dev = deepcopy(config)
        orig_cfg_dev.device = device
        old_cfg_dev = deepcopy(config)
        old_cfg_dev.trainer.device = device
        old_cfg_dev.pop('device', None)

        for cfg_dev in (old_cfg_dev, orig_cfg_dev):
            predictor = PytorchPredictionPipeline(config=cfg_dev, weights=None, device=device)
            model_device = list(predictor.model.parameters())[0].device
            assert model_device == torch.device(device)

    def test_input_from_image_path(self):
        pred_cfg = deepcopy(config)
        vortex_predictor = PytorchPredictionPipeline(config=pred_cfg, weights=None, device='cpu')

        def _test(predictor):
            kwargs = {}
            results = predictor.run(images = ['tests/test_dataset/classification/val/cat/1.jpeg'],
                                    visualize = True,
                                    dump_visual = True,
                                    output_dir = 'tests/output_predict_test',
                                    show_result = False,
                                    **kwargs)
            self._check_result(results)
            self._check_pipeline(predictor)

            # If dump_visualization and images is provided as string, allow for dump visualized image
            vis_dump_path = Path('tests/output_predict_test') / 'prediction_1.jpeg'
            assert vis_dump_path.exists()

            assert all([isinstance(spec, dict) and \
                ('shape' in spec) and ('pos' in spec) and ('type' in spec) and \
                isinstance(spec['shape'], (tuple, list)) and \
                isinstance(spec['type'], str) and \
                isinstance(spec['pos'], int)
                    for spec in predictor.model.input_specs.values()
            ])

        ## normal predictor
        _test(vortex_predictor)

        ## no class names
        vortex_predictor.model.class_names = None
        _test(vortex_predictor)

    @pytest.mark.parametrize("visualize", [
        pytest.param(False, id="no visualize"), 
        pytest.param(True, id="visualize")
    ])
    def test_input_from_numpy(self, visualize):
        # Instantiate predictor
        kwargs = {}
        vortex_predictor = PytorchPredictionPipeline(config = config,
                                                     weights = None,
                                                     device = 'cpu')

        # Read image
        image_data = cv2.imread('tests/test_dataset/classification/val/cat/1.jpeg')
        results = vortex_predictor.run(images = [image_data],
                                       visualize = visualize,
                                       show_result = False,
                                       **kwargs)
        self._check_result(results, visualize=visualize)
        self._check_pipeline(vortex_predictor)

    def test_weight_from_argument(self):
        vortex_predictor = PytorchPredictionPipeline(config=config,
                                                     weights=pth_model_path,
                                                     device = 'cpu')
        image_data = cv2.imread('tests/test_dataset/classification/val/cat/1.jpeg')
        results = vortex_predictor.run(images = [image_data],
                                   visualize = False,
                                   show_result=False)
        self._check_result(results, visualize=False)
        self._check_pipeline(vortex_predictor)


@pytest.mark.parametrize(
    "weight", [
        pytest.param(None, id="default weight"),
        pytest.param(pth_model_path, id="custom weight path")
    ]
)
def test_export_pipeline(weight):
    exported_paths = [
        Path(config.output_directory).joinpath(config.experiment_name, "{}.onnx".format(config.experiment_name)),
        Path(config.output_directory).joinpath(config.experiment_name, "{}_bs8.onnx".format(config.experiment_name)),
        Path(config.output_directory).joinpath(config.experiment_name, "{}.pt".format(config.experiment_name)),
        Path(config.output_directory).joinpath(config.experiment_name, "{}_bs8.pt".format(config.experiment_name))
    ]

    # Initialize graph exporter
    graph_exporter = GraphExportPipeline(config=config, weights=weight)

    status = graph_exporter.run(example_input=None)
    assert isinstance(status,EasyDict)
    assert 'export_status' in status.keys()
    assert isinstance(status.export_status, bool)
    assert all(p.exists() for p in exported_paths)

    ## without class names
    graph_exporter.class_names = None
    status = graph_exporter.run(example_input=None)
    assert isinstance(status,EasyDict)
    assert 'export_status' in status.keys()
    assert isinstance(status.export_status, bool)
    assert all(p.exists() for p in exported_paths)

    ckpt = torch.load(pth_model_path)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    assert state_dict_is_equal(state_dict, graph_exporter.predictor.model.state_dict())


class TestIRValidationPipeline():

    @pytest.mark.parametrize("batch_size", [
        pytest.param(1, id="single batch"),
        pytest.param(8, id="multi batch")
    ])
    def test_onnx_validation(self, batch_size):
        suffix = "{}.onnx".format('_bs8' if batch_size == 8 else '')
        model_path = Path(config.output_directory) / config.experiment_name / (config.experiment_name + suffix)

        validation_executor = IRValidationPipeline(config=config, model=model_path,
            backends = 'cpu', generate_report = True)

        eval_results = validation_executor.run(batch_size=batch_size)

        # Check return value
        assert isinstance(eval_results,EasyDict)

        # Check generated reports
        report_dir = Path(config.output_directory) / config.experiment_name / 'reports'
        generated_report = Path(report_dir) / '{}_onnx_IR_validation_cpu.md'.format(config.experiment_name)
        assert generated_report.exists()

    @pytest.mark.parametrize("batch_size", [
        pytest.param(1, id="single batch"),
        pytest.param(8, id="multi batch")
    ])
    def test_torchscript_validation(self, batch_size):
        suffix = "{}.pt".format('_bs8' if batch_size == 8 else '')
        model_path = Path(config.output_directory) / config.experiment_name / (config.experiment_name + suffix)

        validation_executor = IRValidationPipeline(config=config,
                                                model = model_path,
                                                backends = 'cpu',
                                                generate_report = True)

        eval_results = validation_executor.run(batch_size=batch_size)

        # Check return value
        assert isinstance(eval_results,EasyDict)

        # Check generated reports
        report_dir = Path(config.output_directory) / config.experiment_name / 'reports'
        generated_report = Path(report_dir) / '{}_torchscript_IR_validation_cpu.md'.format(config.experiment_name)
        assert generated_report.exists()

class TestIRPredictionPipeline(): 

    @pytest.mark.parametrize(
        "model_input", [
            pytest.param(onnx_model_path, id="onnx"),
            pytest.param(pt_model_path, id="torchscript")
        ]
    )
    def test_model_api(self, model_input):
        vortex_ir_predictor = IRPredictionPipeline(model=model_input, runtime='cpu')
        
        assert isinstance(vortex_ir_predictor.model.input_specs,OrderedDict)
        assert isinstance(vortex_ir_predictor.model.class_names,list)

    @pytest.mark.parametrize(
        "model_input", [
            pytest.param(onnx_model_path, id="onnx"),
            pytest.param(pt_model_path, id="torchscript")
        ]
    )
    def test_input_from_image_path(self, model_input):
        # Instantiate predictor
        kwargs = {}
        vortex_ir_predictor = IRPredictionPipeline(model=model_input, runtime='cpu')

        results = vortex_ir_predictor.run(
            images = ['tests/test_dataset/classification/val/cat/1.jpeg'],
            visualize = True, dump_visual = True,
            output_dir = 'tests/output_predict_test',
            show_result = False,
            **kwargs
        )

        # Prediction pipeline output must be EasyDict
        assert isinstance(results,EasyDict)
        assert 'prediction' in results.keys()
        assert 'visualization' in results.keys()

        # Prediction output muSt be in a list -> representation of batched output list index [0] means result for input [0]
        assert isinstance(results.prediction,list)
        assert isinstance(results.visualization,list)

        # Check list member type
        assert isinstance(results.prediction[0],EasyDict)
        assert isinstance(results.visualization[0],np.ndarray)

        # If dump_visualization and images is provided as string, allow for dump visualized image
        vis_dump_path = None
        if model_input==onnx_model_path:
            vis_dump_path = Path('tests/output_predict_test') / 'onnx_ir_prediction_1.jpeg'
        elif model_input==pt_model_path:
            vis_dump_path = Path('tests/output_predict_test') / 'torchscript_ir_prediction_1.jpeg'
        assert vis_dump_path is not None and vis_dump_path.exists()

    @pytest.mark.parametrize(
        "model_input", [
            pytest.param(onnx_model_path, id="onnx"),
            pytest.param(pt_model_path, id="torchscript")
        ]
    )
    def test_input_from_numpy_with_vis(self, model_input):
        # Instantiate predictor
        kwargs = {}
        vortex_ir_predictor = IRPredictionPipeline(model = model_input,
                                                runtime = 'cpu')
        
        # Read image
        image_data = cv2.imread('tests/test_dataset/classification/val/cat/1.jpeg')

        results = vortex_ir_predictor.run(images = [image_data],
                                      visualize = True,
                                      show_result = False,
                                      **kwargs)
        
        # Prediction pipeline output must be EasyDict
        assert isinstance(results,EasyDict)
        assert 'prediction' in results.keys()
        assert 'visualization' in results.keys()

        # Prediction output muSt be in a list -> representation of batched output list index [0] means result for input [0]
        assert isinstance(results.prediction,list)
        assert isinstance(results.visualization,list)

        # Check list member type
        assert isinstance(results.prediction[0],EasyDict)
        assert isinstance(results.visualization[0],np.ndarray)

    @pytest.mark.parametrize(
        "model_input", [
            pytest.param(onnx_model_path, id="onnx"),
            pytest.param(pt_model_path, id="torchscript")
        ]
    )
    def test_input_from_numpy_wo_vis(self,model_input):
        # Instantiate predictor
        kwargs = {}
        vortex_ir_predictor = IRPredictionPipeline(model=model_input, runtime='cpu')

        # Read image
        image_data = cv2.imread('tests/test_dataset/classification/val/cat/1.jpeg')

        results = vortex_ir_predictor.run(
            images = [image_data], visualize = False,
            show_result = False, **kwargs
        )
        
        # Prediction pipeline output must be EasyDict
        assert isinstance(results,EasyDict)
        assert 'prediction' in results.keys()
        assert 'visualization' in results.keys()

        # Prediction output muSt be in a list -> representation of batched output list index [0] means result for input [0]
        assert isinstance(results.prediction,list)
        assert results.visualization is None

        # Check list member type
        assert isinstance(results.prediction[0],EasyDict)


class TestHypOptPipeline:

    def test_train_obj(self):

        hypopt = HypOptPipeline(config=config,optconfig=hypopt_train_obj_config)
        trial_result = hypopt.run()

        dump_report_path = Path(config.output_directory) / config.experiment_name / 'hypopt' / hypopt_train_obj_config.study_name / 'best_params.txt'

        assert isinstance(trial_result,EasyDict)
        assert 'best_trial' in trial_result.keys()
        assert dump_report_path.exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
    def test_train_obj_gpu(self):
        config.trainer.device = 'cuda'
        hypopt = HypOptPipeline(config=config,optconfig=hypopt_train_obj_config)
        trial_result = hypopt.run()

        dump_report_path = Path(config.output_directory) / config.experiment_name / 'hypopt' / hypopt_train_obj_config.study_name / 'best_params.txt'

        assert isinstance(trial_result,EasyDict)
        assert 'best_trial' in trial_result.keys()
        assert dump_report_path.exists()

    # TODO add validation objective test
    def test_val_obj(self):
        pass

    def test_remove_output(self):
        ## remove test output
        ## TODO: have a better implementation with fixtures
        shutil.rmtree("tests/output_test/test_classification_pipelines")
        shutil.rmtree("tests/output_predict_test")
