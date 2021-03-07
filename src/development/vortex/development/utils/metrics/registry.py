import pytorch_lightning as pl
from vortex.development.utils.registry import Registry

METRICS = Registry(name='metrics', base_class=pl.metrics.Metric)