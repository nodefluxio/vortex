import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from vortex.development.networks.models import ModelBase


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

