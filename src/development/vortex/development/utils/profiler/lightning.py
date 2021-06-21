import pytorch_lightning as pl
from .speed import TimeData
from .resource import GPUMonitor, CPUMonitor
from ..metrics import MetricBase
from typing import List
from pathlib import Path
from .resource import get_uname, get_cpu_info, get_cpu_scaling, get_gpu_info

class MarkdownGen:
    def __init__(self):
        super().__init__()
        self.doc = ''
    
    @classmethod
    def make_table(cls, header: List[str], data: List[List[str]]):
        # simply accepts matrix
        table = ""
        table += "|" + "|".join(header) + "|\n"
        table += "|" + "|".join(["---"]*len(header)) + "|\n"
        for row, rows in enumerate(data):
            table += "|" + "|".join(rows) + "|\n"
        return table
    
    @classmethod
    def make_lists(cls, data: List[str]):
        lists = "\n".join(map(lambda s: f"- {s}", data))
        return lists
    
    @classmethod
    def make_image(cls, name: str, path: str, title=""):
        img = f"![{name}]({path} \"{title}\")"
        return img
    
    def write(self, texts: str):
        self.doc += f"{texts}\n"
    
    def add_section(self, title: str, texts=""):
        self.doc += f"# {title}\n"
        self.doc += f"{texts}\n"
    
    def add_subsubsection(self, title: str, texts=""):
        self.doc += f"## {title}\n"
        self.doc += f"{texts}\n"
    
    def save(self, output_filename: str):
        return Path(output_filename).write_text(self.doc)

class Profiler(pl.profiler.profilers.BaseProfiler):
    def __init__(self, plot_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_dir = plot_dir
        self.timers = {}
        self._init_resource_monitor()
        self._start_resource_monitor()
    
    def _init_resource_monitor(self):
        self.cpu_monitor = CPUMonitor(name='global_cpu_monitor')
        self.gpu_monitor = GPUMonitor(name='global_gpu_monitor')
    
    def _start_resource_monitor(self):
        self.cpu_monitor.start()
        self.gpu_monitor.start()

    def start(self, action_name: str):
        if action_name not in self.timers:
            self.timers[action_name] = TimeData(action_name)
        self.timers[action_name].start()
    
    def stop(self, action_name: str):
        self.timers[action_name].stop()
    
    def summary(self, plot_dir=None) -> str:
        self.cpu_monitor.stop()
        self.gpu_monitor.stop()
        report_str = []
        resource_plots = []
        plot_dir = plot_dir or self.plot_dir
        for action_name, time_data in self.timers.items():
            action_report = time_data.report()
            action_report = f'{action_report["mean"]} (mean); {action_report["median"]} (median);'
            action_result = f'{action_name} : {action_report}'
            report_str.append(action_result)
            if time_data.name == 'runtime_call':
                runtime_call_outputs = time_data.plot(plot_dir)
                for field_name, path in runtime_call_outputs.items():
                    report_str.append(f'{field_name}: {str(path)}')
                    resource_plots.append((field_name,str(path)))
        # for now just plot cpu & gpu monitor
        if plot_dir:
            cpu_monitor_outputs = self.cpu_monitor.plot(plot_dir)
            gpu_monitor_outputs = self.gpu_monitor.plot(plot_dir)
            for field_name, path in cpu_monitor_outputs.items():
                report_str.append(f'{field_name}: {str(path)}')
                resource_plots.append((field_name,str(path)))
            for field_name, path in gpu_monitor_outputs.items():
                report_str.append(f'{field_name}: {str(path)}')
                resource_plots.append((field_name,str(path)))
        self.report_str = report_str
        self.resource_plots = resource_plots
        return '\n'.join(report_str)
    
    def report(self, trainer=None, model=None, output_directory='.', experiment_name='reports'):
        if model is not None and hasattr(model, 'metrics'):
            metric = model.metrics
            output_directory = Path(output_directory)
            if isinstance(metric, MetricBase):
                md = MarkdownGen()

                tolist = lambda x: [x[0], str(x[1])]
                metric_results = metric.compute()
                metric_results = list(map(tolist, metric_results.items()))
                metric_results = md.make_table(['metric name', 'value'], metric_results)
                md.add_section('Metrics', metric_results)

                toimage = lambda x: md.make_image(x[0], x[1], x[0])
                metric_assets = metric.report(output_directory, experiment_name)
                metric_assets = list(map(toimage, metric_assets.items()))
                metric_assets = md.make_lists(metric_assets)
                md.add_section('Assets', metric_assets)

                resource_plots = self.resource_plots
                resource_plots = list(map(toimage,resource_plots))
                resource_plots = md.make_lists(resource_plots)
                md.add_section('Resources', resource_plots)

                f = lambda x: f'{x[0]}: {x[1]}'
                environment = dict(
                    uname=get_uname(),
                    cpu_info=get_cpu_info(),
                    cpu_scaling=get_cpu_scaling(),
                    gpu_info=get_gpu_info(),
                )
                environment = list(map(f, environment.items()))
                environment = md.make_lists(environment)
                md.add_section('Environment', environment)

                return md
