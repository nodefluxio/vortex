import torch
import warnings
import torch.nn as nn
import torch.optim as optim
from . import lr_scheduler
import inspect
from inspect import Signature, Parameter
from collections import OrderedDict
from typing import Tuple, List, Union, Type, Any, Dict
import math

from vortex.development.utils import type_utils
from vortex.development.utils.logger.base_logger import ExperimentLogger

class BaseTrainer(object):
    __loss_parameters__ = ['input', 'target']
    def __init__(self, model: Type[nn.Module], optimizer: Union[Type[optim.Optimizer], Tuple[str, Dict], Dict[str,Any]], 
                 scheduler: Union[Type[optim.lr_scheduler._LRScheduler], Tuple[str, Dict[str, Any]], Dict[str,Any]], 
                 criterion: Type[nn.Module], experiment_logger : Union[Type[ExperimentLogger],None] = None,check_annotations: bool = True):

        self.model = model
        self.optimizer = type(self).create_optimizer(optimizer, self.model)
        self.scheduler = type(self).create_scheduler(scheduler, self.optimizer)
        self.criterion = criterion
        self.experiment_logger = experiment_logger
        self.global_step = 0

        self._check_model()
    
    def _check_model(self, strict=False, check_annotations=True):
        """
        check model and loss, called after model, loss, optim, scheduler are assigned
        can be bypassed from derived class by overriding this method
        strict mode can be set from derived class by overriding this method and 
        pass strict=True
        """
        model_signature = inspect.signature(self.model.forward)
        loss_signature = inspect.signature(self.criterion.forward)
        criterion_args = type(self).__loss_parameters__

        ## force loss fn signature has parameter named 'input' and 'targets'
        args_exist = all(arg in loss_signature.parameters for arg in criterion_args)
        model_return_anno = model_signature.return_annotation
        args_not_exist_msg = "loss function {} does not have {} args".format(
            type(self.criterion), ', '.join(criterion_args)
        )
        ## assuming first name of loss param is input name
        loss_input_name = criterion_args[0]
        loss_input_anno = loss_signature.parameters[loss_input_name].annotation \
            if loss_input_name in loss_signature.parameters else None
        anno_mismatch_msg = "model return annotation does not match with loss input annotation, \
            {model_anno} with {loss_anno}".format_map(dict(
                model_anno=model_return_anno,
                loss_anno=loss_input_anno,
        ))
        def raise_error(cond, msg, exc=TypeError):
            if not cond:
                raise exc(msg)
        def emit_warning(cond, msg):
            if not cond:
                warnings.warn(msg)
        warn_or_error = raise_error if strict else emit_warning
        warn_or_error(args_exist,args_not_exist_msg)
        if not check_annotations:
            return
        return_type_annotated = model_return_anno != Signature.empty
        loss_input_annotated = loss_input_anno is not None
        warn_or_error(return_type_annotated, "return type not annotated")
        warn_or_error(loss_input_annotated, "loss input not annotated")
        match = return_type_annotated and loss_input_annotated and type_utils.match_annotation(model_return_anno,loss_input_anno)
        warn_or_error(match, "annotation mismatch")
            
    
    @staticmethod
    def create_optimizer(optimizer: Union[optim.Optimizer, tuple, dict], model: Type[nn.Module]):
        """
        create optimizer
        """
        if isinstance(optimizer, tuple):
            warnings.warn("creating optimizer from tuple is deprecated", PendingDeprecationWarning)
            assert (len(optimizer) == 2), "expect length of optimizer is 2 if type of tuple"
            assert isinstance(optimizer[0], str), "expect optimizer is type of Tuple[str,Dict], got optimizer[0] : %s" % (
                type(optimizer[0]))
            assert isinstance(optimizer[1], dict), "expect optimizer is type of Tuple[str,Dict], got optimizer[0] : %s" % (
                type(optimizer[1]))
            optimizer = dict(
                module=optimizer[0],
                args=optimizer[1],
            )
        if isinstance(optimizer, dict):
            if 'method' in optimizer and not 'module' in optimizer:
                optimizer.update({'module': optimizer['method']})
            opt_method, kwargs = optimizer['module'], optimizer['args']
            if opt_method.startswith('optim.'):
                opt_method = opt_method.replace('optim.','')
            assert hasattr(optim, opt_method), \
                "unsupported optimizer {}".format(opt_method)
            kwargs.update({'params' : model.parameters()})
            optimizer = getattr(optim, opt_method)(**kwargs)
        return optimizer

    @staticmethod
    def create_scheduler(scheduler, optimizer):
        """
        create scheduler
        """
        if isinstance(scheduler, dict):
            if 'method' in scheduler and not 'module' in scheduler:
                scheduler.update({'module': scheduler['method']})
            sch_method, kwargs = scheduler['module'], scheduler['args']
            assert hasattr(lr_scheduler, sch_method), \
                "unsupported lr_scheduler {}".format(sch_method)
            kwargs.update({'optimizer': optimizer})
            scheduler = getattr(lr_scheduler, sch_method)(**kwargs)

            # Assign scheduler type update type
            scheduler.step_update = None
            for step_update_type in lr_scheduler.step_update_map:
                if type(scheduler).__name__ in lr_scheduler.step_update_map[step_update_type]:
                    scheduler.step_update = step_update_type
            if scheduler.step_update == None:
                raise RuntimeError('Currently, scheduler {} is not supported, please select other scheduler'.format(type(scheduler).__name__))
        return scheduler
    
    @staticmethod
    def apply_scheduler_step(scheduler,epoch,step,steps_per_epoch,accumulation_step):
        """ Apply scheduler step
        """
        if scheduler.step_update == 'batch_update':
            try:
                scheduler.step()
            except:
                scheduler.step(epoch)
        elif scheduler.step_update == 'epoch_update':
            accumulated_step = math.floor(step / accumulation_step )
            accumulated_steps_per_epoch = math.floor(steps_per_epoch / accumulation_step)
            if accumulated_step == accumulated_steps_per_epoch -1:
                try:
                    scheduler.step()
                except:
                    scheduler.step(epoch)

        

    def train(self, dataloader, epoch: int):
        raise NotImplementedError
    
    def __call__(self, dataloader, epoch: int):
        is_training = self.model.training
        self.model.train()
        train_results = self.train(dataloader, epoch)
        self.model.train(is_training)
        return train_results