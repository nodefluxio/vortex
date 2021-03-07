import torch
import pytorch_lightning as pl
from vortex.runtime import create_runtime_model
from .metrics import METRICS

class RuntimeWrapper(pl.LightningModule):
    def __init__(self, path, runtime='cpu', val_args={}, metric_args={}, profiler=None):
        super().__init__()
        # TODO: pass additional val args
        self.val_args = val_args
        self.metric_args = metric_args

        # create runtime model
        self.model = create_runtime_model(path, runtime)

        # get metrics name from model, then retrieve from registry
        metrics = self.model.properties['vortex.metrics']
        metrics = metrics.replace('[','').replace(']','').split(',')
        # take first metric for now
        # TODO: construct all metrics
        self.metrics = METRICS.create_from_args(metrics[0], **metric_args)

        # setup profiler
        self.profiler = profiler or pl.profiler.PassThroughProfiler()
        # infer batch size from model, assume the first input is image input
        self.batch_size = self.model.input_specs['input']['shape'][0]
    
    def forward(self, *args, **kwargs):
        val_args = self.val_args
        to_numpy = lambda x: x.numpy()
        args = list(map(to_numpy, args))
        kwargs = dict(map(lambda k, v: (k,to_numpy(v)),kwargs))
        with self.profiler.profile('runtime_call'):
            results =  self.model(*args, **kwargs, **val_args)
        return results
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        fmt = self.model.output_format
        self.metrics.update(y_hat, y)
    
    def test_epoch_end(self, validation_step_outputs):
        self.log_dict(self.metrics.compute())