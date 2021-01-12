import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pathlib import Path
from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader

from vortex.development.networks.models import ModelBase
from vortex.development.core.trainer import TrainingPipeline
from vortex.development import __version__ as vortex_version


_REQUIRED_TRAINER_CFG = {
    'experiment_name': 'dummy_experiment',
    'device': 'cuda:0',
    'trainer': {
        'optimizer': {
            'method': 'SGD',
            'args': {'lr': 0.001}
        },
        'epoch': 2
    }
}

class DummyDataset(Dataset):
    def __init__(self, num_classes=5, data_size=224, num_data=5):
        super().__init__()

        self.num_classes = num_classes
        self.num_data = num_data
        self.data_size = data_size

    def __getitem__(self, index: int):
        x = torch.randn(3, self.data_size, self.data_size)
        y = torch.randint(0, self.num_classes, (1,))[0]
        return x, y

class DummyModel(ModelBase):
    def __init__(self, num_classes=5):
        super().__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16, self.num_classes)

        self.accuracy = pl.metrics.Accuracy()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc(x)
        return x

    def predict(self, x: torch.Tensor):
        x = self(x)
        conf, label = x.softmax(1).max(1)
        return torch.stack((label.float(), conf), dim=1)

    @property
    def output_format(self):
        return {
            "class_label": {"indices": [0], "axis": 0},
            "class_confidence": {"indices": [1], "axis": 0}
        }

    @property
    def available_metrics(self):
        return {
            'accuracy': 'max',
            'train_loss': 'min'
        }

    def training_step(self, batch, batch_idx):
        data, target = batch
        pred = self(data)
        loss = self.criterion(pred, target)
        self.log('train_loss', loss, on_epoch=True, on_step=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self(data)
        acc = self.accuracy(pred.detach().cpu(), target.detach().cpu())
        self.log('accuracy', acc, on_epoch=True)


def patched_pl_trainer(experiment_dir, model):
    TrainingPipeline._patch_trainer_components()
    trainer = pl.Trainer(default_root_dir=experiment_dir)
    TrainingPipeline._patch_trainer_object(trainer)

    trainer.accelerator_backend = trainer.accelerator_connector.select_accelerator()
    trainer.accelerator_backend.setup(model)
    trainer.accelerator_backend.train_loop = trainer.train
    trainer.accelerator_backend.validation_loop = trainer.run_evaluation
    trainer.accelerator_backend.test_loop = trainer.run_evaluation
    return trainer

## TODO: test logger


@pytest.mark.parametrize(
    ('device', 'expected_gpu', 'expected_auto_select'),
    [
        pytest.param(None, None, False, id="on cpu"),
        pytest.param("cuda", 1, True, id="on gpu autoselect"),
        pytest.param("cuda:1", "1", False, id="on gpu 1")
    ]
)
def test_decide_device(device, expected_gpu, expected_auto_select):
    config = dict(device=device)
    expected = dict(gpus=expected_gpu, auto_select_gpus=expected_auto_select)

    kwargs = TrainingPipeline._decide_device_to_use(config)
    assert kwargs == expected


def test_handle_validation_interval():
    val_epoch = 1

    config = dict(validator=dict(val_epoch=val_epoch))
    expected = dict(check_val_every_n_epoch=val_epoch)
    kwargs = TrainingPipeline._handle_validation_interval(config)
    assert kwargs == expected

    config = dict(trainer=dict(validate_interval=val_epoch))
    kwargs = TrainingPipeline._handle_validation_interval(config)
    assert kwargs == expected

    with pytest.raises(RuntimeError):
        config = dict(validator=dict(val_epoch="1,2"))
        TrainingPipeline._handle_validation_interval(config)

    with pytest.raises(RuntimeError):
        config = dict(trainer=dict(validate_interval="1,2"))
        TrainingPipeline._handle_validation_interval(config)


## TODO; test checkpoint
def test_checkpoint_default_last(tmp_path: Path):
    config = deepcopy(_REQUIRED_TRAINER_CFG)
    config = EasyDict(config)

    num_classes = 5
    model = DummyModel(num_classes=num_classes)
    model.config = config
    model.class_names = ["label_"+str(n) for n in range(num_classes)]
    trainer = patched_pl_trainer(str(tmp_path), model)

    ## dummy metrics data
    metrics = {
        'train_loss': 1.0891,
        'accuracy': 0.9618
    }
    trainer.logger_connector.callback_metrics = metrics

    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(str(tmp_path), config, model)
    ckpt_callbacks[0].on_pretrain_routine_start(trainer, model)
    ckpt_callbacks[0].on_validation_end(trainer, model)

    fname = config['experiment_name'] + "-last.pth"
    fpath = tmp_path.joinpath("version_0", "checkpoints", fname)
    assert fpath.exists()

    ## check saved checkpoint
    checkpoint = torch.load(fpath)
    assert checkpoint['config'] == dict(config)
    assert checkpoint['metrics'] == metrics
    assert checkpoint['class_names'] == model.class_names
    assert checkpoint['vortex_version'] == vortex_version
    assert 'checkpoint_last' in checkpoint['callbacks']


def test_checkpoint_save_best():
    pass


def test_checkpoint_save_epoch():
    pass


## TODO: test metrics

## TODO: test created model

## TODO: other vortex behavior
