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
        """Added to behaves like dict

        Args:
            key (str): key

        Returns:
            type: type associated with "key"
        """
        return self.get(key)
    
    def register_module(self, module, name: Union[str,None]=None, force: bool=False, overwrite: bool=False):
        # typecheck
        if not isinstance(force,bool):
            raise TypeError("expect force to be string type, got {}".format(type(force)))
        # typecheck
        if not isinstance(overwrite,bool):
            raise TypeError("expect overwrite to be string type, got {}".format(type(overwrite)))
        # typecheck
        # if not (inspect.isfunction(module) or inspect.isclass(module)):
        if not inspect.isclass(module):
            raise TypeError("expect module to be a type or function")
        # only accepts:
        # - base_class is None: happily accepts any class
        # - force is True: acts like base_class is None
        # - base_class is not None: module must be a subclass of base_class

        # set local "base_class" to None if forced, hence acts like self.base_class is None
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
        """Alias to register_module
        """
        self.register_module(*args, **kwargs)
    
    def pop(self, module: str, *args):
        """Remove desired class associated with `module`

        Args:
            module (str): key

        Returns:
            Any: popped module
        """
        return self.modules.pop(module,*args)
    
    def keys(self):
        """List all keys to registered modules

        Returns:
            list of str: registered module's key
        """
        return self.modules.keys()
    
    def values(self):
        """List all values (class) to registered modules

        Returns:
            list of type: registered module's type
        """
        return self.modules.values()
    
    def items(self):
        """Key value pairs to registered modules

        Returns:
            list of pairs of (str,type): kv pairs
        """
        return self.modules.items()
    
    def copy(self):
        """Added to behaves like dict

        Returns:
            self: copy self
        """
        return copy(self)

    def register(self, name: str=None, force: bool=False, overwrite: bool=False):
        """Decorator Factory to register module

        Args:
            name (str, optional): desired name of the module. Defaults to None.
            force (bool, optional): force the decorated class to be added, may ignore base_class. Defaults to False.
            overwrite (bool, optional): overwrite existing module if exists. Defaults to False.

        Returns:
            Callable: decorator function
        """
        # actualy decorator factor, bind args to be called
        def _register(cls):
            self.register_module(module=cls,name=name,force=force,overwrite=overwrite)
            return cls
        return _register
    
    def create_from_args(self, module: str, *args, **kwargs):
        """create instance from positional/keyword args

        Args:
            module (str): module's key in which to instantiated

        Returns:
            object: instance of registered module associated with "module"
        """
        cls = self[module]
        return cls(*args, **kwargs)
    
    def create_from_dict(self, module: str, args: dict):
        """create instance from args packed as dictionary

        Args:
            module (str): module's key in which to instantiated
            args (dict): args to be passed

        Returns:
            object: instance of registered module associated with "module"
        """
        # don't pass `module` as keyword args
        return self.create_from_args(module, **args)