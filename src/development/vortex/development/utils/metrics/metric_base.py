import pytorch_lightning as pl
import abc

class MetricBase(pl.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
    
    @abc.abstractmethod
    def report(self) -> dict:
        pass