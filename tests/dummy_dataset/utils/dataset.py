import numpy as np
from torch.utils.data import Dataset

supported_dataset = [
    'DummyDataset'
]

class DummyDataset(Dataset):
    def __init__(self, total=10, **kwargs):
        self.kwargs = kwargs
        self.data_format = {
            'class_label': None
        }
        self.class_names = ['0','1','2','3','4','5','6','7','8','9']

        self.images = np.random.randn(total, 224, 224, 3)
        self.targets = np.random.randint(0, 10, size=total)
        self.total = total
    
    def __getitem__(self, idx):
        return self.images[idx], int(self.targets[idx])
    
    def __len__(self):
        return self.total


def create_dataset(**kwargs):
    return DummyDataset(**kwargs)
