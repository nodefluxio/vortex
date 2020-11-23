import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from time import time
from pathlib import Path
from typing import Union
from contextlib import ContextDecorator

class TimeData(ContextDecorator):
    def __init__(self, name):
        self.name = name
        self.data = []
        self.n_calls = 0
        self.total_time = 0
        self.t0 = None
        self.t1 = None

    def update(self, dt):
        self.n_calls += 1
        self.total_time += dt
        self.data.append(dt)
    
    def __enter__(self) :
        self.t0 = time()
        return self
    
    def __exit__(self, *exc) :
        self.t1 = time()
        self.update(self.t1 - self.t0)
    
    def report(self) :
        results = dict(
            mean=np.mean(self.data),
            median=np.median(self.data),
            quantile=np.quantile(self.data,q=[0.25,0.75]),
            percentile=np.percentile(self.data,q=np.arange(100)),
        )
        return results

    @staticmethod
    def plot_time_and_percentile(results : dict, filename : str, unit : str) :
        plt.cla()
        plt.clf()
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(2)
        plt.gcf().set_size_inches((6.4,9.6))
        ax1.set_title("timedata")
        ax2.set_title("percentile")
        ax1.set(xlabel='iteration')
        ax1.set(ylabel='time ({})'.format(unit))
        ax2.set(xlabel='n-th percentile')
        ax2.set(ylabel='time ({})'.format(unit))
        p = np.arange(100)
        percentile, data = results['percentile'], results['data']
        median, mean = results['median'], results['mean']
        mean_x = np.abs(percentile-mean).argmin()
        p_90 = percentile[90]
        timedata = pd.DataFrame(data, index=range(len(data)), columns=["timedata"])
        sns.lineplot(data=timedata, palette="tab10", linewidth=0.15, ax=ax1)
        ax2.plot(p, percentile,label='percentile')
        ax2.scatter(
            50, median, linestyle=':', 
            label='median ({0:.2f})'.format(median)
        )
        ax2.scatter(
            90, p_90, linestyle=':', 
            label='90-th percentile ({0:.2f})'.format(p_90)
        )
        ax2.scatter(
            np.abs(percentile-mean).argmin(),
            mean, label='mean ({0:.2f})'.format(mean)
        )
        ax2.legend()
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(filename)
        plt.gcf().set_size_inches((6.4,4.8)) ## reset back to matplotlib default
        sns.reset_defaults()
    
    def plot(self, output_directory, filename: Union[str]=None, unit:str='ms') :
        unit_map = dict(ms=lambda x: x * 1e3, s=lambda x: x, fps=lambda x: 1./x)
        results_s = self.report()
        assert unit in unit_map
        u = unit_map[unit]
        results = {key : u(value) for key, value in results_s.items()}
        results['data'] = u(np.asarray(self.data))
        output_directory = Path(output_directory)
        output_filename = output_directory / '{}.png'.format(self.name)
        TimeData.plot_time_and_percentile(
            results=results, filename=output_filename, unit=unit
        )
        return {'timedata' : output_filename}
