from torch.utils.data.dataloader import DataLoader

supported_loaders = [('PytorchDataLoader','default')] # (Supported data loader, Supported dataset wrapper format)

def create_loader(*args,**kwargs):
    return DataLoader(*args,**kwargs)