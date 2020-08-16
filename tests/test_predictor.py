import os
import sys
from pathlib import Path
proj_path = os.path.abspath(Path(__file__).parents[1])
sys.path.append(proj_path)
sys.path.append('src/runtime')
sys.path.append('src/development')

import torch
import torch.nn as nn
import numpy as np
import pytest

from vortex.development.predictor import create_predictor
from vortex.development.predictor.base_module import BasePredictor
from vortex.development.networks import create_model_components
from vortex.development.utils.parser.parser import load_config, check_config
from vortex.development.predictor import get_prediction_results


torch.manual_seed(123)

def test_base_predictor():
    output_format = {
        "classification": {
            "class_label": {"indices": [0], "axis": 1},
            "class_confidence": {"indices": [1], "axis": 1}
        },
        "detection": {
            "bounding_box": {"indices": [0,1,2,3], "axis": 1},
            "class_label": {"indices": [5], "axis": 1},
            "class_confidence": {"indices": [4], "axis": 1}
        }
    }
    model = nn.Linear(3, 5)
    for task, f in output_format.items():
        model.task = task
        model.output_format = f
        predictor = BasePredictor(model)
        out = torch.randn(1, 3, 224, 224)

        assert not predictor.training, "predictor needs to be in eval mode"
        assert predictor.output_format == f

@pytest.mark.parametrize("task", ["classification", "detection"])
def test_predictor(task):
    config_path = os.path.join(proj_path, "tests", "config", "test_" + task + ".yml")
    config = load_config(config_path)
    check_result = check_config(config, experiment_type='train')
    assert check_result.valid, "config file %s for task %s is not valid, "\
        "result:\n%s" % (config_path, task, str(check_result))
    
    args = {
        'model_name': config.model.name,
        'preprocess_args': config.model.preprocess_args,
        'network_args': config.model.network_args,
        'loss_args': config.model.loss_args,
        'postprocess_args': config.model.postprocess_args,
        'stage': 'train'
    }
    model_components = create_model_components(**args)
    predictor = create_predictor(model_components)
    assert not predictor.training
    assert hasattr(predictor, "output_format")
    assert predictor.model.task == task

    s = config.model.preprocess_args.input_size
    x = torch.randint(0, 256, size=(1, s, s, 3))
    args = {}
    if task == 'detection':
        args["score_threshold"] = torch.tensor([config.validator.args["score_threshold"]], dtype=torch.float32)
        args["iou_threshold"] = torch.tensor([config.validator.args["iou_threshold"]], dtype=torch.float32)
    
    with torch.no_grad():
        out = predictor(x, **args)
    out_np = np.asarray(out)
    if out_np.size != 0:
        result = get_prediction_results(out_np, predictor.output_format)[0]
        for name, val in result.items():
            indices = predictor.output_format[name]['indices']
            assert val.shape[-1] == len(indices), f"output_format for {name} could not be "\
                f"retrieved properly, size missmatch expect {len(indices)} got {val.shape[-1]}"

    num_elem = sum(len(f["indices"]) for f in predictor.output_format.values())
    assert out[0].size(-1) == num_elem, "number of element in output of predictor is not "\
        "the same as described in output_format; expect %s got %s" % (num_elem, out.size(1))


if __name__ == "__main__":
    test_base_predictor()
    for task in ("detection", "classification"):
        test_predictor(task)
