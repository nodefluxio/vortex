"""
test case for core trainer
"""
import sys
sys.path.append('src/development')
sys.path.append('src/runtime')


import pytest
import torch
import torch.nn as nn

import vortex.development.core.engine as engine
from vortex.development.core.engine import create_trainer
from vortex.development.core.engine.trainer.default_trainer import DefaultTrainer
from vortex.development.core.engine.trainer.base_trainer import ExperimentLogger, BaseTrainer

experiment_logger = ExperimentLogger()

class BrokenDummyLossFN(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(BrokenDummyLossFN, self).__init__(*args, **kwargs)
    def forward(self, some_input, some_targets, another_args) :
        return 10

class DummyLossFN(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(DummyLossFN, self).__init__(*args, **kwargs)
    def forward(self, input : torch.Tensor, target : torch.Tensor) :
        return 10

class DummyModel(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(DummyModel, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(1,2)
    def forward(self, input : torch.Tensor) -> torch.Tensor :
        return self.fc(input)

class BrokenDummyModel(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(BrokenDummyModel, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(1,2)
    def forward(self, input : torch.Tensor) :
        return self.fc(input)

class BrokenCustomTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(type(self),*args, **kwargs)

class CustomTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(type(self),self).__init__(*args, **kwargs)
    
    def train(self, dataset, epoch):
        ## dummy train
        return 0

class StrictCustomTrainer(DefaultTrainer):
    def __init__(self, *args, **kwargs):
        super(type(self),self).__init__(*args, **kwargs)
    
    def _check_model(self):
        ## enable strict mode from base trainer
        super(type(self),self)._check_model(strict=True)

    def train(self, dataset, epoch):
        ## dummy train
        return 0

class AnnotatedL1Loss(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super(type(self),self).__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor, target):
        return super(type(self),self).forward(input=input,target=target)

optimizer = ( 'SGD', {
    'lr' : 1e-3,
    'momentum' : 0.9,
})
scheduler = ( 'StepLR', {
    'step_size' : 10,
})

def test_construct_from_tuple() :
    model = DummyModel()
    loss_fn = DummyLossFN()
    trainer = DefaultTrainer(
        optimizer=optimizer, 
        scheduler=scheduler, 
        model=model, 
        criterion=loss_fn, 
        experiment_logger=experiment_logger
    )

def test_custom_trainer():
    loss_fn = nn.L1Loss()
    model = DummyModel()
    cfg = dict(
        driver=dict(
            module='UnregisteredTrainer',
            args={},
        )
    )
    with pytest.raises(RuntimeError) as e:
        trainer = create_trainer(cfg, 
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
        )
    
    cfg = dict(
        driver=dict(
            module='CustomTrainer',
            args={},
        )
    )
    engine.register_trainer(CustomTrainer)
    trainer = create_trainer(cfg, 
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=loss_fn,
    )
    engine.remove_trainer('CustomTrainer')
    with pytest.raises(RuntimeError) as e:
        trainer = create_trainer(cfg, 
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
        )

def test_strict_custom_trainer():
    loss_fn = AnnotatedL1Loss()
    model = DummyModel()
    
    engine.register_trainer(StrictCustomTrainer)
    cfg = dict(
        driver=dict(
            module='StrictCustomTrainer',
            args={},
        )
    )
    trainer = create_trainer(cfg, 
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=loss_fn,
    )
    with pytest.raises(TypeError) as e:
        model = BrokenDummyModel()
        trainer = create_trainer(cfg, 
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
        )
    engine.remove_trainer('StrictCustomTrainer')

def test_broken_construct_from_tuple() :
    with pytest.warns(UserWarning) as e:
        ## default trainer only warns
        model = BrokenDummyModel()
        loss_fn = DummyLossFN()
        trainer = DefaultTrainer(
            optimizer=optimizer, 
            scheduler=scheduler, 
            model=model, 
            criterion=loss_fn, 
            experiment_logger=experiment_logger
        )
    with pytest.raises(TypeError) as e:
        model = BrokenDummyModel()
        loss_fn = DummyLossFN()
        trainer = StrictCustomTrainer(
            optimizer=optimizer, 
            scheduler=scheduler, 
            model=model, 
            criterion=loss_fn, 
            experiment_logger=experiment_logger
        )

loss_fn = nn.L1Loss()
model = DummyModel()

def test_broken_construct_from_tuple_2() :
    with pytest.warns(UserWarning) as e:
        ## default trainer only warns
        loss_fn = BrokenDummyLossFN()
        trainer = DefaultTrainer(
            optimizer=optimizer, 
            scheduler=scheduler, 
            model=model, 
            criterion=loss_fn, 
            experiment_logger=experiment_logger
        )
    
    with pytest.raises(TypeError) as e:
        ## using strict trainer raises type error
        loss_fn = BrokenDummyLossFN()
        trainer = StrictCustomTrainer(
            optimizer=optimizer, 
            scheduler=scheduler, 
            model=model, 
            criterion=loss_fn, 
            experiment_logger=experiment_logger
        )

def test_torch_nn_loss() :
    trainer = DefaultTrainer(
        optimizer=optimizer, 
        scheduler=scheduler, 
        model=model, 
        criterion=loss_fn, 
        experiment_logger=experiment_logger
    )

optimizer = dict(
    module='SGD',
    args=dict(
        lr=1e-3,
        momentum=0.9,
    )
)
scheduler = dict(
    module='StepLR',
    args=dict(
        step_size=10,
    )
)

def test_construct_from_dict():
    trainer = DefaultTrainer(
        optimizer=optimizer, 
        scheduler=scheduler, 
        model=model, 
        criterion=loss_fn, 
        experiment_logger=experiment_logger
    )

def test_create_trainer():
    cfg = dict(
        driver=dict(
            module='DefaultTrainer',
            args={},
        )
    )
    trainer = create_trainer(cfg, 
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=loss_fn,
    )

## TODO : test case with trainer __call__, dummy dataset
