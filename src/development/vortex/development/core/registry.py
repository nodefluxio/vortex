from typing import Dict, Any, Union

class Registry:
    def __init__(self, name: str, base_class=None):
        self.base_class = base_class
        self.name = name
        self.modules: Dict[str,Any] = {}
    
    def get(self, key):
        return self.modules[key]
    
    def __len__(self):
        return len(self.modules)
    
    def __contains__(self, key: str):
        return key in self.modules
    
    def __repr__(self):
        cls_name = self.__class__.__name__
        fmt = f"{cls_name}<{self.name}>"
        return fmt
    
    def __getitem__(self, key: str):
        return self.get(key)
    
    def register_module(self, module: type, name: Union[str,None]=None, force: bool=False):
        if not isinstance(module, type):
            raise TypeError("expect module to be a type")
        if self.base_class is not None:
            if not issubclass(module, self.base_class):
                raise TypeError(
                    "expect module to be a subclass of {}, got {}"
                    .format(self.base_class.__name__,module.__name__)
                )
        if name is None:
            name = module.__name__
        if not force and name in self.modules:
            raise KeyError(
                "module {} already exists, use force=True to override"
                .format(name)
            )
        self.modules[name] = module
    
    def register(self, name: str=None, force: bool=False):
        # actualy decorator factor, bind args to be called
        def _register(cls):
            self.register_module(module=cls,name=name,force=force)
            return cls
        return _register
    
    def create_from_args(self, module: str, *args, **kwargs):
        # create instance from positional/keyword args
        cls = self[module]
        return cls(*args, **kwargs)
    
    def create_from_dict(self, module: str, args: dict):
        # create instance from args packed as dictionary
        return self.create_from_args(module, **args)