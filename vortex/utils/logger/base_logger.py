from pprint import PrettyPrinter
import logging
import random
import string
import warnings

pp = PrettyPrinter(indent=4)

class ExperimentLogger():
    def __init__(self,config = None):
        self.log_on_hyperparameters(config=config)

        def get_run_key():
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))

        self.run_key=get_run_key()

    def log_on_hyperparameters(self,config):
        warnings.warn("Function 'log_on_hyperparameters' is not implemented!!")

    def log_on_step_update(self,metrics_log):
        warnings.warn("Function 'log_on_step_update' is not implemented!!")

    def log_on_epoch_update(self,metrics_log):
        warnings.warn("Function 'log_on_epoch_update' is not implemented!!")

    def log_on_model_save(self,file_log):
        warnings.warn("Function 'log_on_model_save' is not implemented!!")

    def log_on_validation_result(self,metrics_log):
        warnings.warn("Function 'log_on_validation_result' is not implemented!!")
