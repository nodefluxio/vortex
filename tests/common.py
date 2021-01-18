import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from vortex.development.networks.models import ModelBase
from vortex.development.pipelines.trainer import TrainingPipeline


MINIMAL_TRAINER_CFG = {
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

    def __len__(self) -> int:
        return self.num_data

    @property
    def class_names(self):
        return ["label_"+str(n) for n in range(self.num_classes)]

class DummyPLDatset(pl.LightningDataModule):

    def __init__(self, batch_size=2, num_classes=5, data_size=224):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.data_size = data_size

    def train_dataloader(self):
        dataset = DummyDataset(self.num_classes, self.data_size)
        return DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader()


class DummyModel(ModelBase):
    def __init__(self, num_classes=5):
        super().__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, self.num_classes)

        self.accuracy = pl.metrics.Accuracy()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
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


def patched_pl_trainer(experiment_dir, model, callbacks=[], trainer_args={}, gpus=True):
    TrainingPipeline._patch_trainer_components()
    if gpus:
        gpus = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        gpus=gpus,
        default_root_dir=experiment_dir,
        callbacks=callbacks,
        **trainer_args
    )
    TrainingPipeline._patch_trainer_object(trainer)

    ## setup accelerator
    trainer.accelerator_backend = trainer.accelerator_connector.select_accelerator()
    trainer.accelerator_backend.setup(model)
    trainer.accelerator_backend.train_loop = trainer.train
    trainer.accelerator_backend.validation_loop = trainer.run_evaluation
    trainer.accelerator_backend.test_loop = trainer.run_evaluation

    ## dummy metrics data
    metrics = {
        'train_loss': torch.tensor(1.0891),
        'accuracy': torch.tensor(0.7618)
    }
    trainer.logger_connector.callback_metrics = metrics
    model.trainer = trainer
    return trainer

def prepare_model(config, num_classes=5):
    model = DummyModel(num_classes=num_classes)
    model.config = config
    model.class_names = ["label_"+str(n) for n in range(num_classes)]
    return model
