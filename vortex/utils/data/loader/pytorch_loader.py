from torch.utils.data.dataloader import DataLoader

supported_loaders = ['PytorchDataLoader']

def create_loader(*args,**kwargs):
    return DataLoader(*args,**kwargs)