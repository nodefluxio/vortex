from comet_ml import Experiment
from vortex.development.utils.logger.base_logger import ExperimentLogger
from flatten_dict import flatten
from easydict import EasyDict

class CometMLLogger(ExperimentLogger):
    def __init__(self, provider_args: EasyDict, config,**kwargs):
        self.experiment=Experiment(
            api_key=provider_args.api_key,
            project_name=provider_args.project_name,
            workspace=provider_args.workspace,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        super().__init__(config)
        self.run_key=self.experiment.get_key()
        self.log_url=self.experiment.url

    def log_on_hyperparameters(self, config: EasyDict):
        hyper_params = {}
        if config is not None:
            hyper_params['model'] = config.model
            hyper_params['trainer'] = config.trainer
            hyper_params['augmentations'] = config.dataset.train.augmentations
        self.experiment.log_parameters(flatten(hyper_params, reducer='path'))

    def log_on_step_update(self,metrics_log : dict):
        step=metrics_log['step']
        metrics_log.pop('step')
        self.experiment.log_metrics(metrics_log,step=step)

    def log_on_epoch_update(self,metrics_log : dict):
        epoch=metrics_log['epoch']
        metrics_log.pop('epoch')
        self.experiment.log_metrics(metrics_log,epoch=epoch)

    def log_on_model_save(self,file_log : dict):
        pass

    def log_on_validation_result(self,metrics_log : dict):
        epoch=metrics_log['epoch']
        metrics_log.pop('epoch')
        self.experiment.log_metrics(metrics_log,epoch=epoch)


def create_logger(provider_args:EasyDict, config, **kwargs):
    return CometMLLogger(provider_args, config, **kwargs)
