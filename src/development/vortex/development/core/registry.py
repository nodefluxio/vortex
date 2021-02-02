from typing import Dict, Any, Union
from copy import copy
import inspect

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
    
    def register_module(self, module, name: Union[str,None]=None, force: bool=False, overwrite: bool=False):
        # typecheck
        if not isinstance(force,bool):
            raise TypeError("expect force to be string type, got {}".format(type(force)))
        # typecheck
        if not isinstance(overwrite,bool):
            raise TypeError("expect overwrite to be string type, got {}".format(type(overwrite)))
        # typecheck
        if not (inspect.isfunction(module) or inspect.isclass(module)):
            raise TypeError("expect module to be a type or function")
        # only accepts:
        # - base_class is None: happily accepts either class or function
        # - force is True: acts like base_class is None
        # - base_class is not None: module must be a subclass of base_class
        base_class = self.base_class if not force else None
        if base_class is not None:
            if not issubclass(module, base_class):
                raise TypeError(
                    "expect module to be a subclass of {}, got {}, use force=True to force register"
                    .format(base_class.__name__,module.__name__)
                )
        # name validation:
        # - module is class: can infer name from class
        # - function: possibly ugly name, name must be explicitly provided
        valid_name = inspect.isclass(module) or isinstance(name,str)
        if not valid_name:
            raise ValueError("name must be provided if module is function")
        if name is None:
            name = module.__name__
        if not overwrite and name in self.modules:
            raise KeyError(
                "module {} already exists, use overwrite=True to overwrite"
                .format(name)
            )
        self.modules[name] = module
    
    def add(self, *args, **kwargs):
        self.register_module(*args, **kwargs)
    
    def pop(self, module: str, *args):
        return self.modules.pop(module,*args)
    
    def keys(self):
        return self.modules.keys()
    
    def values(self):
        return self.modules.values()
    
    def items(self):
        return self.modules.items()
    
    def copy(self):
        return copy(self)

    def register(self, name: str=None, force: bool=False, overwrite: bool=False):
        # actualy decorator factor, bind args to be called
        def _register(cls):
            self.register_module(module=cls,name=name,force=force,overwrite=overwrite)
            return cls
        return _register
    
    def create_from_args(self, module: str, *args, **kwargs):
        # create instance from positional/keyword args
        cls = self[module]
        return cls(*args, **kwargs)
    
    def create_from_dict(self, module: str, args: dict):
        # create instance from args packed as dictionary
        return self.create_from_args(module, **args)