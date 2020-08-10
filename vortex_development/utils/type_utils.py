from typing import Union, Tuple, List
from multipledispatch import dispatch

## python 3.8 -> typing.get_origin(model_type), typing.get_args(model_type)
def get_origin(annotation):
    if hasattr(annotation, '__origin__'):
        return annotation.__origin__
    else:
        return type(annotation)

def get_args(annotation) -> Tuple[type]:
    if get_origin(annotation) in (Union, Tuple, List):
        return annotation.__args__
    elif isinstance(annotation, type):
        return tuple(annotation)
    else:
        raise TypeError("unsupported")

@dispatch(list, type)
def _match_annotation(lhs, rhs):
    return rhs in lhs

@dispatch(type, list)
def _match_annotation(lhs, rhs):
    return _match_annotation(rhs, lhs)

@dispatch(list, list)
def _match_annotation(lhs, rhs):
    return any(r in lhs for r in rhs)

@dispatch(type, type)
def _match_annotation(lhs, rhs):
    return lhs == rhs

def match_annotation(lhs, rhs):
    if get_origin(lhs) in (Union,):
        lhs = list(get_args(lhs))
    if get_origin(rhs) in (Union,):
        rhs = list(get_args(rhs))
    return _match_annotation(lhs, rhs)

    