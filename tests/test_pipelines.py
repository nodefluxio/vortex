import os
import sys
from pathlib import Path
proj_path = os.path.abspath(Path(__file__).parents[1])
sys.path.append(proj_path)

import shutil
from easydict import EasyDict
import pytest
import numpy as np
import cv2
import yaml
import torch

from vortex.core.pipelines import (
    TrainingPipeline,
    PytorchValidationPipeline,
    PytorchPredictionPipeline,
    IRValidationPipeline,
    GraphExportPipeline,
    IRPredictionPipeline,
    HypOptPipeline
)
from vortex.utils.parser.parser import load_config
from vortex.utils.parser.loader import Loader


config_path = 'tests/config/test_classification_pipelines.yml'
hypopt_train_obj_path = 'tests/config/test_hypopt_train_objective.yml'
onnx_model_path = 'tests/output_test/test_classification_pipelines/test_classification_pipelines.onnx'
pt_model_path = 'tests/output_test/test_classification_pipelines/test_classification_pipelines.pt'


# Load configuration from experiment file
config = load_config(config_path)
with open(hypopt_train_obj_path) as f:
    hypopt_train_obj_config = EasyDict(yaml.load(f, Loader=Loader))


def test_train_pipeline():

    # Clear pre-existing output file
    if Path(config.output_directory).exists():
        shutil.rmtree(Path(config.output_directory))

    if Path('experiments/local_runs.log').exists():
        os.remove(Path('experiments/local_runs.log'))

    # Instantiate Training
    train_executor = TrainingPipeline(config=config,config_path=config_path,hypopt=False)
    output = train_executor.run()

    # Check output type
    assert isinstance(output,EasyDict)

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
    assert len(val_metrics) == int(config.trainer.epoch/config.trainer.validation.val_epoch)
    assert isinstance(learning_rates,list)
    assert len(learning_rates) == config.trainer.epoch

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
    assert ckpt['config'] == config
    assert tuple(ckpt['class_names']) == ('cat', 'dog')


def test_validation_pipeline():

    # Instantiate Validation
    validation_executor = PytorchValidationPipeline(config=config,
                                                 weights = None,
                                                 backends = 'cpu',
                                                 generate_report = True)

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

    def test_input_from_image_path(self):
        # Instantiate predictor
        kwargs = {}
        vortex_predictor = PytorchPredictionPipeline(config = config,
                                        weights = None,
                                        device = 'cpu')

        def _test(predictor):
            results = predictor.run(images = ['tests/images/cat.jpg'],
                                    visualize = True,
                                    dump_visual = True,
                                    output_dir = 'tests/output_predict_test',
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

            # If dump_visualization and images is provided as string, allow for dump visualized image
            vis_dump_path = Path('tests/output_predict_test') / 'prediction_cat.jpg'
            assert vis_dump_path.exists()
        
        ## normal predictor
        _test(vortex_predictor)

        ## no class names
        vortex_predictor.class_names = None
        _test(vortex_predictor)

    def test_input_from_numpy_with_vis(self):
        # Instantiate predictor
        kwargs = {}
        vortex_predictor = PytorchPredictionPipeline(config = config,
                                                     weights = None,
                                                     device = 'cpu')

        # Read image
        image_data = cv2.imread('tests/images/cat.jpg')
        results = vortex_predictor.run(images = [image_data],
                                   visualize = True,
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

    def test_input_from_numpy_wo_vis(self):
        # Instantiate predictor
        kwargs = {}
        vortex_predictor = PytorchPredictionPipeline(config = config,
                                        weights = None,
                                        device = 'cpu')

        # Read image
        image_data = cv2.imread('tests/images/cat.jpg')
        results = vortex_predictor.run(images = [image_data],
                                   visualize = False,
                                   **kwargs)

        # Prediction pipeline output must be EasyDict
        assert isinstance(results,EasyDict)
        assert 'prediction' in results.keys()
        assert 'visualization' in results.keys()

        # Prediction output muSt be in a list -> representation of batched output list index [0] means result for input [0]
        assert isinstance(results.prediction,list)
        assert results.visualization is None

        # Check list member type
        assert isinstance(results.prediction[0],EasyDict)

def test_export_pipeline():
    # Initialize graph exporter
    graph_exporter = GraphExportPipeline(config=config, weights=None)

    status = graph_exporter.run(example_input=None)
    assert isinstance(status,EasyDict)
    assert 'export_status' in status.keys()
    assert isinstance(status.export_status, bool)

    ## without class names
    graph_exporter.class_names = None
    status = graph_exporter.run(example_input=None)
    assert isinstance(status,EasyDict)
    assert 'export_status' in status.keys()
    assert isinstance(status.export_status, bool)

class TestIRValidationPipeline():

    def test_onnx_validation_single_batch(self):
        # Instantiate Validation
        model_path = Path(config.output_directory) / config.experiment_name / (config.experiment_name +'.onnx')

        validation_executor = IRValidationPipeline(config=config,
                                                model = model_path,
                                                backends = 'cpu',
                                                generate_report = True)

        eval_results = validation_executor.run(batch_size=1)

        # Check return value
        assert isinstance(eval_results,EasyDict)

        # Check generated reports
        report_dir = Path(config.output_directory) / config.experiment_name / 'reports'
        generated_report = Path(report_dir) / '{}_onnx_IR_validation_cpu.md'.format(config.experiment_name)
        assert generated_report.exists()
    
    def test_onnx_validation_multi_batch(self):
        # Instantiate Validation
        model_path = Path(config.output_directory) / config.experiment_name / (config.experiment_name +'_bs2.onnx')

        validation_executor = IRValidationPipeline(config=config,
                                                model = model_path,
                                                backends = 'cpu',
                                                generate_report = True)

        eval_results = validation_executor.run(batch_size=2)

        # Check return value
        assert isinstance(eval_results,EasyDict)

        # Check generated reports
        report_dir = Path(config.output_directory) / config.experiment_name / 'reports'
        generated_report = Path(report_dir) / '{}_onnx_IR_validation_cpu.md'.format(config.experiment_name)
        assert generated_report.exists()

    def test_torchscript_validation_single_batch(self):
        # Instantiate Validation
        model_path = Path(config.output_directory) / config.experiment_name / (config.experiment_name +'.pt')

        validation_executor = IRValidationPipeline(config=config,
                                                model = model_path,
                                                backends = 'cpu',
                                                generate_report = True)

        eval_results = validation_executor.run(batch_size=1)

        # Check return value
        assert isinstance(eval_results,EasyDict)

        # Check generated reports
        report_dir = Path(config.output_directory) / config.experiment_name / 'reports'
        generated_report = Path(report_dir) / '{}_torchscript_IR_validation_cpu.md'.format(config.experiment_name)
        assert generated_report.exists()
    
    def test_torchscript_validation_multi_batch(self):
        # Instantiate Validation
        model_path = Path(config.output_directory) / config.experiment_name / (config.experiment_name +'_bs2.pt')

        validation_executor = IRValidationPipeline(config=config,
                                                model = model_path,
                                                backends = 'cpu',
                                                generate_report = True)

        eval_results = validation_executor.run(batch_size=2)

        # Check return value
        assert isinstance(eval_results,EasyDict)

        # Check generated reports
        report_dir = Path(config.output_directory) / config.experiment_name / 'reports'
        generated_report = Path(report_dir) / '{}_torchscript_IR_validation_cpu.md'.format(config.experiment_name)
        assert generated_report.exists()

class TestIRPredictionPipeline(): 

    @pytest.mark.parametrize("model_input", [onnx_model_path,pt_model_path])
    def test_input_from_image_path(self,model_input):
        # Instantiate predictor
        kwargs = {}
        vortex_ir_predictor = IRPredictionPipeline(model = model_input,
                                                runtime = 'cpu')

        results = vortex_ir_predictor.run(images = ['tests/images/cat.jpg'],
                                      visualize = True,
                                      dump_visual = True,
                                      output_dir = 'tests/output_predict_test',
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

        # If dump_visualization and images is provided as string, allow for dump visualized image

        if model_input==onnx_model_path:
            vis_dump_path = Path('tests/output_predict_test') / 'onnx_ir_prediction_cat.jpg'
        elif model_input==pt_model_path:
            vis_dump_path = Path('tests/output_predict_test') / 'torchscript_ir_prediction_cat.jpg'
        assert vis_dump_path.exists()

    @pytest.mark.parametrize("model_input", [onnx_model_path,pt_model_path])
    def test_input_from_numpy_with_vis(self,model_input):
        # Instantiate predictor
        kwargs = {}
        vortex_ir_predictor = IRPredictionPipeline(model = model_input,
                                                runtime = 'cpu')
        
        # Read image
        image_data = cv2.imread('tests/images/cat.jpg')

        results = vortex_ir_predictor.run(images = [image_data],
                                      visualize = True,
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

    @pytest.mark.parametrize("model_input", [onnx_model_path,pt_model_path])
    def test_input_from_numpy_wo_vis(self,model_input):
        # Instantiate predictor
        kwargs = {}
        vortex_ir_predictor = IRPredictionPipeline(model = model_input,
                                                runtime = 'cpu')
        
        # Read image
        image_data = cv2.imread('tests/images/cat.jpg')

        results = vortex_ir_predictor.run(images = [image_data],
                                      visualize = False,
                                      **kwargs)
        
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
    
    #TODO add test_validation_obj
        