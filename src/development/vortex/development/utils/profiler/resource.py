import matplotlib.pyplot as plt
import re
import time
import psutil
import threading
import gpustat
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path


def get_uname() :
    import subprocess
    result = subprocess.run(['uname', '-a'], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8")

def get_cpu_info() :
    import subprocess
    result = subprocess.run(['lscpu'], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8")

def get_cpu_scaling() :
    import subprocess
    def get_scaling_gov(cpu : int) :
        cpu = '/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor'.format(int(cpu))
        result = subprocess.run(['cat', cpu], stdout=subprocess.PIPE)
        return result.stdout.decode("utf-8").rstrip()
    nproc = int(subprocess.run(['nproc'], stdout=subprocess.PIPE).stdout)
    scaling = [get_scaling_gov(i) for i in range(nproc)]
    return scaling

def get_gpu_info() :
    import subprocess
    # result = subprocess.run("lspci | grep ' NVIDIA ' | grep ' VGA ' | cut -d" " -f 1 | xargs -i lspci -v -s {}".split(' '), shell=True, stdout=subprocess.PIPE)
    p0 = subprocess.Popen(('lspci'), stdout=subprocess.PIPE)
    p1 = subprocess.Popen(('grep', " NVIDIA "), stdin=p0.stdout, stdout=subprocess.PIPE)
    p0.stdout.close()
    p2 = subprocess.Popen(('grep', " VGA "), stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(('cut', '--delimiter= ', '-f', '1'), stdin=p2.stdout, stdout=subprocess.PIPE)
    p2.stdout.close()
    out = subprocess.check_output(('xargs', '-i', 'lspci', '-v', '-s', '{}'), stdin=p3.stdout)
    p3.wait()
    return out.decode("utf-8")

class CPUMonitor(threading.Thread) :
    def __init__(self, name, dt=0.5, *args, **kwargs) :
        self._args, self._kwargs = args, kwargs
        super(CPUMonitor, self).__init__(*args, **kwargs)
        self.name = name
        self.cpu_percent_data = []
        self.cv = threading.Condition()
        self.running = False
        self.dt = dt

    def clone(self) :
        return CPUMonitor(self.name, self.dt, *self._args, **self._kwargs)
    
    def stop(self) :
        with self.cv :
            self.running = False
    
    def run(self) :
        self.running = True
        running = True
        while running :
            cpu_percent = psutil.cpu_percent(interval=self.dt, percpu=True)
            self.cpu_percent_data.append(cpu_percent)
            with self.cv :
                running = self.running
    
    def report(self) :
        results = dict(
            cpu_percent=self.cpu_percent_data,
        )
        return results
    
    def plot(self, output_directory, *args, **kwargs) :
        results = self.report()
        output_directory = Path(output_directory)
        output_directory = Path(output_directory)
        output_filename = output_directory / '{}.png'.format(self.name)
        plt.cla()
        plt.clf()
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(2)
        plt.gcf().set_size_inches((6.4,9.6))
        cpu_data_array = np.asarray(self.cpu_percent_data)
        n_cpu = 1 if len(cpu_data_array.shape)==1 else cpu_data_array.shape[-1]
        columns = ['cpu{}'.format(i) for i in range(n_cpu)] if n_cpu > 1 else ['cpu']
        linewidth = 0.75
        cpu_data = pd.DataFrame(
            cpu_data_array, columns=columns
        )
        sns.lineplot(data=cpu_data, palette="tab10", linewidth=linewidth, dashes=False, ax=ax1)
        ax1.set(xlabel='time (x{0:.2f}s)'.format(self.dt), ylabel='Utilization (%)')
        ax1.set_title("CPU Utilization (%)")

        ax2.set_title("CPU Utilization (%)")
        ax2.boxplot(cpu_data_array, showfliers=False)
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(output_filename)

        plt.gcf().set_size_inches((6.4,4.8)) ## reset back to matplotlib default
        sns.reset_defaults()
        
        return {'cpu_percent' : output_filename}

class GPUMonitor(threading.Thread) :
    ## TODO : capture keyboard interrupt
    def __init__(self, name, dt=0.1, process_regex='python*', *args, **kwargs) :
        self._args, self._kwargs = args, kwargs
        super(GPUMonitor,self).__init__(*args, **kwargs)
        self._kwargs.update(dict(process_regex=process_regex))
        self.utilization = []
        self.memory = []
        self.temperature = []
        self.cv = threading.Condition()
        self.running = False
        self.name = name
        self.dt = dt
        self.process_memory = {}
        self.process_regex = re.compile(process_regex)
    
    def clone(self) :
        return GPUMonitor(self.name, self.dt, *self._args, **self._kwargs)

    def stop(self) :
        with self.cv :
            self.running = False
    
    def run(self) :
        self.running = True
        running = True
        while running :
            try :
                gpu = gpustat.core.GPUStatCollection.new_query()
            except gpustat.core.N.NVMLError :
                """
                OSError: libnvidia-ml.so.1: cannot open shared object file: No such file or directory
                """
                ## TODO : emit warning
                gpu = None
            if not gpu is None :
                ## TODO : support multi-gpu monitoring
                utilization = gpu[0].utilization
                memory = gpu[0].memory_used
                temperature = gpu[0].temperature
                self.utilization.append(utilization)
                self.memory.append(memory)
                self.temperature.append(temperature)
                processes = gpu[0].processes
                ## filter process by regex
                f = lambda x : self.process_regex.match(x['command'])
                processes = filter(f, processes)
                for process in processes :
                    pid = process['pid']
                    name = process['command']
                    memory = process['gpu_memory_usage']
                    memory = [*self.process_memory[pid]['memory'], memory] if pid in self.process_memory else [memory]
                    process_memory = dict(name=name, memory=memory)
                    self.process_memory.update({pid : process_memory})
            time.sleep(self.dt)
            with self.cv :
                running = self.running

    def report(self) :
        process_memory = self.process_memory
        process_memory = [dict(name=process['name'],memory=np.mean(process['memory'])) for pid, process in process_memory.items()]
        results = dict(
            gpu_utilization=self.utilization,
            gpu_memory_usage=self.memory,
            process_memory=process_memory,
        )
        # print(self.process_memory)
        return results
    
    def plot(self, output_directory, *args, **kwargs) :
        if not len(self.utilization) :
            return {}
        results = self.report()
        output_directory = Path(output_directory)
        output_directory = Path(output_directory)
        output_filenames = {}
        ## plot gpu usage
        output_filename = output_directory / '{}_utilization.png'.format(self.name)
        plt.cla()
        plt.clf()
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(2)
        plt.gcf().set_size_inches((6.4,9.6))
        gpu_data = np.asarray(self.utilization)
        linewidth, columns = 0.5, ['utilization']
        gpu_data = pd.DataFrame(
            gpu_data, columns=columns
        )
        sns.lineplot(data=gpu_data, palette="tab10", linewidth=linewidth, dashes=False, ax=ax1)
        ax1.set_title("GPU Utilization (%)")
        ax1.set(xlabel='time (x{0:.2f}s)'.format(self.dt), ylabel='Utilization (%)')

        ax2.set_title("GPU Utilization (%)")
        percentile = np.percentile(gpu_data, q=np.arange(100))
        ax2.boxplot(percentile, showfliers=False)
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        output_filenames.update(dict(gpu_utilization=output_filename))

        ## plot gpu usage
        output_filename = output_directory / '{}_temperature.png'.format(self.name)
        plt.cla()
        plt.clf()
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(2)
        plt.gcf().set_size_inches((6.4,9.6))
        gpu_data = np.asarray(self.temperature)
        linewidth, columns = 0.5, ['temperature']
        gpu_data = pd.DataFrame(
            gpu_data, columns=columns
        )
        sns.lineplot(data=gpu_data, palette="tab10", linewidth=linewidth, dashes=False, ax=ax1)
        ax1.set_title("GPU Temperature")
        ax1.set(xlabel='time (x{0:.2f}s)'.format(self.dt), ylabel='Temperature')

        ax2.set_title("GPU Temperature")
        percentile = np.percentile(gpu_data, q=np.arange(100))
        ax2.boxplot(percentile, showfliers=False)
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        output_filenames.update(dict(gpu_temperature=output_filename))

        ## plot gpu memory
        output_filename = output_directory / '{}_memory.png'.format(self.name)
        plt.cla()
        plt.clf()
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(2)
        plt.gcf().set_size_inches((6.4,9.6))
        gpu_data = {}
        for key, memory in self.process_memory.items() :
            memory_used = memory['memory']
            field_name = '{} ({})'.format(memory['name'], np.mean(memory_used))
            columns.append(field_name)  # e.g. python (500), python3.5 (1024)
            gpu_data[field_name] = memory_used
        gpu_data['system'] = np.asarray(self.memory)
        # make sure values in gpu_data have same length, so pandas won't complains
        n_data = min(map(lambda v: len(v), gpu_data.values()))
        for key, value in gpu_data.items():
            gpu_data[key] = value[:n_data]
        linewidth = 0.5
        gpu_df = pd.DataFrame(
            data=gpu_data
        )
        sns.lineplot(data=gpu_df, palette="tab10", linewidth=linewidth, dashes=False, ax=ax1)
        ax1.set(xlabel='time (x{0:.2f}s)'.format(self.dt), ylabel='GPU Memory')
        ax1.set_title("GPU Memory")

        ax2.set_title("GPU Memory")
        ax2.boxplot(gpu_data.values(), showfliers=False)
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(output_filename)
        output_filenames.update(dict(gpu_memory=output_filename))

        plt.gcf().set_size_inches((6.4,4.8)) ## reset back to matplotlib default
        sns.reset_defaults()

        return output_filenames