import inspect
from functools import partial


supported_models = {}

def _register_model(module, name=None, force=False):
    """
    Register model with model_name.
    Args:
        model_name: str
        m: module or function, should have `create_model_components`
            function, or is named `create_model_components`
    """
    global supported_models

    if not (inspect.isfunction(module) or inspect.isclass(module)):
        raise RuntimeError("module ({}) must be a function or a class".format(module))
    if name is None:
        name = module.__name__

    ## check if exists
    if not force and name in supported_models:
        raise RuntimeError("Model name '{}' is already registered. Check that you register "
            "the correct module, or use other name, or use argument 'force=True'.".format(name))
    supported_models[name] = module
    return module

def register_model(name=None, module=None, force=False):
    if not isinstance(force, bool):
        raise TypeError(f"'force' argument must be boolean. got {type(force)}")

    ## use it as regular function -> register_model('name', module)
    if module is not None:
        return _register_model(module, name, force)

    ## use as decorator but not calling the function -> @register_model
    if inspect.isclass(name) or inspect.isfunction(name):
        module, name = name, None
        return _register_model(module, name, force)

    if not (name is None or isinstance(name, str)):
        raise TypeError("'name' argument must be an 'str', got {}".format(type(name)))

    ## use it as a decorator -> @register_model()
    return partial(_register_model, name=name, force=force)

def remove_model(model_name):
    global supported_models
    if not model_name in supported_models:
        return False

    supported_models.pop(model_name)
    return True
