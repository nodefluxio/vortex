from torch.utils.data.dataloader import DataLoader

supported_loaders = [('PytorchDataLoader','default')]

def create_loader(*args,**kwargs):
    return DataLoader(*args,**kwargs)