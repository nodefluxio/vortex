import torch
import logging
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from copy import copy
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
from easydict import EasyDict
from collections import OrderedDict
from collections.abc import Sequence
from functools import singledispatch
from typing import Union, List, Dict, Type, Any

from vortex.predictor.base_module import BasePredictor, create_predictor
from vortex.predictor.utils import get_prediction_results

from vortex.utils.profiler.speed import TimeData
from vortex.utils.profiler.resource import CPUMonitor, GPUMonitor
from vortex.core.factory import create_runtime_model
# from vortex.core.pipelines.prediction_pipeline import IRPredictionPipeline

from vortex_runtime.basic_runtime import BaseRuntime

## set High DPI for matplotlib
## TODO: properly set image dpi
matplotlib.rcParams['figure.dpi'] = 125

def no_collate(batch) :
    ## don't let pytorch default collater to try stacking targets
    images = list(map(lambda x: x[0], batch))
    targets = list(map(lambda x: x[1], batch))
    return images, targets

class Logger:
    """
    logger wrapper with callable fn and default log level
    """
    def __init__(self, logger, default_log_level='debug'):
        log_level = ['debug', 'info', 'warn', 'warning', 'error', 'fatal']
        assert all(hasattr(logger, level) for level in log_level)
        assert default_log_level in log_level
        self.logger = logger
        import types
        self.log = types.MethodType(getattr(self.logger, default_log_level), self)
    
    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

class ResourceMonitorWrapper(object):
    """
    resource monitor wrapper enabling `with` syntax
    """
    def __init__(self, monitors):
        self.monitors = monitors
    def __enter__(self):
        for i in range(len(self.monitors)):
            ## stop if monitor is running
            if self.monitors[i].is_alive() :
                self.monitors[i].stop()
            ## reset if monitor is already started so thread can be restarted
            if self.monitors[i].ident :
                self.monitors[i] = self.monitors[i].clone()
            ## actually start or restart the thread
            self.monitors[i].start()
        return self
    def __exit__(self, type, value, traceback):
        for i in range(len(self.monitors)):
            self.monitors[i].stop()

class BaseValidator:
    """
    base class for validation
    """
    def __init__(self, predictor: Union[BasePredictor,BaseRuntime], dataset, experiment_name='validate', output_directory='.', batch_size:int=1,**prediction_args):
        if not isinstance(predictor, (BasePredictor,BaseRuntime)):
            raise RuntimeError("expects `predictor` to have type of BasePredictor or BaseRuntime, " \
                "got %s" % type(predictor))

        self.predictor = predictor
        self.prediction_args = prediction_args
        self.dataset = dataset
        self.metrics = None
        self.experiment_name = experiment_name
        self.output_directory = Path(output_directory)
        self.batch_size = batch_size
        
        self.predictor_name = '{}'.format(self.predictor.__class__.__name__)
        if isinstance(self.predictor, BasePredictor):
            self.predictor_name = '{}[{}]'.format(self.predictor_name, next(self.predictor.parameters()).device)
        if isinstance(predictor, BaseRuntime) :
            predictor_batch_size = predictor.input_specs['input']['shape'][0]
            assert predictor_batch_size == batch_size, \
                "predictor expects batch_size of {}, but got {}".format(
                    predictor_batch_size, batch_size
                )

        self._init_logger()
        self._init_dataset()
        self._init_class_names()
        self._init_profiler()
        self._init_batch()
        self._check_output_format()
    
    def _init_logger(self):
        """
        default logger initialization
        """
        logger = logging.getLogger(type(self).__name__)
        self.logger = Logger(logger)
        self.logger('predictor type : {}'.format(type(self.predictor)))

    def _init_batch(self):
        """
        default batch initialization, convert to dataloader if necessary
        """
        batch_size = self.batch_size
        ## convert to DataLoader, if necessary
        self.dataset = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, collate_fn=no_collate) \
            if batch_size > 1 else self.dataset
    
    def _init_profiler(self):
        """
        default profiler initializer
        """
        experiment_name = self.experiment_name
        self.predict_timedata = TimeData(name='{}_predict_time_{}'.format(experiment_name, self.predictor_name))
        self.cpu_monitor = CPUMonitor(name='{}_cpu_resource_{}'.format(experiment_name, self.predictor_name))
        self.gpu_monitor = GPUMonitor(name='{}_gpu_resource_{}'.format(experiment_name, self.predictor_name))
        self.monitor = ResourceMonitorWrapper([self.cpu_monitor, self.gpu_monitor])
    
    def _init_class_names(self):
        """
        default class names initializer
        """
        if hasattr(self.dataset, 'class_names'):
            class_names = self.dataset.class_names
        elif hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'class_names'):
            class_names = self.dataset.dataset.class_names
        else:
            class_names = None
        self.class_names = class_names
    
    def _init_dataset(self):
        """
        default dataset check, make sure image and label is torch.Tensor or np.ndarray
        """
        if not hasattr(self.dataset, 'data_format'):
            raise RuntimeError("expects dataset to have `data_format` fields explaining " \
                "the dataset return format")
        img, lbl = self.dataset[0]
        if not (isinstance(img, (torch.Tensor, np.ndarray))):
            raise RuntimeError("expects dataset to return `image` of type np.ndarray or " \
                "torch.Tensor, got %s" % type(img))
        if not (isinstance(lbl, (torch.Tensor, np.ndarray))):
            raise RuntimeError("expects dataset to return `label` of type np.ndarray or " \
                "torch.Tensor, got %s" % type(lbl))
        self.labels_fmt = EasyDict(self.dataset.data_format)
        self.result_fmt = EasyDict(self.predictor.output_format)

    def _check_output_format(self):
        """
        default fn to check output format
        """
        ## check for static member variable `__output_format__` from subclass
        if not hasattr(type(self), '__output_format__') : 
            msg = "validator {} doesn't have `__output_format__`".format(type(self))
            # warnings.warn(msg)
            raise TypeError(msg)
        elif not all(fmt in self.result_fmt for fmt in type(self).__output_format__) :
            msg = "missing output format, expects {} got {}".format(type(self).output_format_requirements, self.result_fmt)
            # warnings.warn(msg)
            raise TypeError(msg)
        else:
            ## no problem
            pass
    
    def validation_args(self) -> Dict[str,Any] :
        """
        reports validation args used for this run, 
        override if additional validation args exists
        e.g. score_threshold, iou_threshold etc.
        """
        return dict(
            batch_size=self.batch_size
        )
    
    def eval_init(self, *args, **kwargs) :
        """
        invoked before entering evaluation loop, *args and *kwargs from __call__ are passed to this fn
        """
        pass
    
    def predict(self, image, *args, **kwargs) -> Union[np.ndarray,torch.Tensor,List[Dict[str,np.ndarray]]]:
        """
        default implementation for predict, developer could overrides if specific impl are necessary for specific task
        """
        if isinstance(self.predictor, BasePredictor) :
            results = type(self).torch_predict(
                predictor=self.predictor, 
                image=image, *args, **kwargs,
            )
        else :
            results = type(self).runtime_predict(
                predictor=self.predictor,
                image=image, *args, **kwargs,
            )
        return results
    
    def update_results(self, index : int, results : List[Dict[str,np.ndarray]], targets : Union[np.ndarray,torch.Tensor], last_index : bool) :
        """
        required for each subclass
        update result(s) for possibly-batched image
        """
        raise NotImplementedError

    def compute_metrics(self, *args, **kwargs) -> dict :
        """
        required for each subclass
        """
        raise NotImplementedError
    
    def plot_resource_metrics(self, output_directory=None, filename=None) -> Dict[str,str] : 
        """
        save resource plot, return saved path
        """
        output_directory = self.output_directory if output_directory is None else output_directory
        output_files = {}
        output_files.update(self.predict_timedata.plot(
            output_directory=output_directory,
        ))
        output_files.update(self.cpu_monitor.plot(
            output_directory=output_directory,
        ))
        output_files.update(self.gpu_monitor.plot(
            output_directory=output_directory,
        ))
        return output_files
    
    def resource_usage(self) -> dict:
        """
        report resource usage
        """
        results = {}
        timedata_reports = self.predict_timedata.report()
        gpu_reports = self.gpu_monitor.report()
        results.update({
            'prediction time (mean)' : timedata_reports['mean'],
            'gpu memory' : gpu_reports['process_memory'],
        })
        return results
    
    def save_metrics(self, output_directory) -> dict :
        """
        optional, expects to return dict of filename
        """
        ## TODO : complete docs
        return {}
    
    @staticmethod
    def to_torch_tensor(image : Union[np.ndarray,torch.Tensor,List[torch.Tensor],List[np.ndarray]]) :
        """
        helper function to create torch tensor resulting 4-dim tensor
        """
        ## TODO : complete docs
        if not (isinstance(image,list) or len(image.shape) == 3 or len(image.shape) == 4):
            raise RuntimeError("expects `image` from dataset to be [3,4]-dimensional tensor or list")
        ## first, convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        elif isinstance(image, list):
            m = np if isinstance(image[0], np.ndarray) else torch
            image = m.stack(image,0)
        ## make sure 4-dim tensor
        if len(image.shape) == 3:
            image = image[np.newaxis,:]
        ## make sure NHWC
        if image.shape[1] == 3:
            # convert to hwc, because predictor requires format in hwc, predictor internally change the format
            image = np.transpose(image, (0, 2, 3, 1))
        if not image.shape[3] == 3:
            raise RuntimeError('unexpected error')
        return torch.from_numpy(image)
    
    @classmethod
    @torch.no_grad()
    def torch_predict(cls, predictor: torch.nn.Module, image, *args, **kwargs) :
        """
        helper function for torch, it is up to developer to use this fn
        """
        ## TODO : complete docs
        device = next(predictor.parameters()).device
        image = cls.to_torch_tensor(image)
        image = image.to(device)
        for i in range(len(args)) :
            if isinstance(args[i], np.ndarray) :
                args[i] = torch.from_numpy(args[i])
            args[i] = args[i].to(device)
        for i in kwargs.keys() :
            if isinstance(kwargs[i], np.ndarray) :
                kwargs[i] = torch.from_numpy(kwargs[i])
            kwargs[i] = kwargs[i].to(device)
        return predictor(image, *args, **kwargs)

    @staticmethod
    def runtime_predict(predictor: BaseRuntime, image : Union[List[np.ndarray],np.ndarray], *args, **kwargs) :
        """
        helper function for BaseRuntime, it is up to developer to use this fn
        """
        ## TODO : complete docs
        if isinstance(image, torch.Tensor) :
            image = image.cpu().numpy()
        if isinstance(image, list) :
            image = np.stack(image)
        ## assuming image input is named 'input'
        ## TODO : deduce image input name
        input_shapes = predictor.input_specs['input']['shape']
        input_dim = len(input_shapes)
        image_dim = len(image.shape)
        image = image[np.newaxis,:] \
            if (input_dim - image_dim) else image
        result = predictor(image,*args, **kwargs)
        return result
    
    def format_output(self, results) :
        """
        format output
        """
        if isinstance(results, torch.Tensor) :
            results = results.cpu().numpy()
        if isinstance(results, (np.ndarray, (list, tuple))) \
            and not isinstance(results[0], (dict,OrderedDict)):
            ## first map to cpu/numpy
            results = list(map(lambda x: x.cpu().numpy() if isinstance(x,torch.Tensor) else x, results))
            ## actually perform output formatting
            results = get_prediction_results(results, self.result_fmt)
        assert isinstance(results[0], (dict, OrderedDict)), "result type {} not understood".format(type(results))
        return results

    def __call__(self, *args, **kwargs):
        """
        default validation pipeline
        """
        if isinstance(self.predictor, BasePredictor) :
            is_training = self.predictor.training
            self.predictor.eval()
        self.eval_init(*args, **kwargs)
        with self.monitor as m:
            for index, (image, targets) in tqdm(enumerate(self.dataset), total=len(self.dataset), 
                                            desc=" eval", leave=False):
                with self.predict_timedata :
                    results = self.predict(image=image)
                results = self.format_output(results)
                last_index = False
                if index == len(self.dataset) -1:
                    last_index = True
                self.update_results(
                    index=index,
                    results=results,
                    targets=targets,
                    last_index=last_index
                )
        self.metrics = self.compute_metrics()
        if isinstance(self.predictor, BasePredictor) :
            self.predictor.train(is_training)
        return self.metrics