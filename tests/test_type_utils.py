import sys
sys.path.append('vortex/development_package')

import pytest
import torch
import numpy as np
from typing import Union, Tuple, List

from vortex.development.utils import type_utils


class DummyClass:
    pass

def test_match_annotation():
    assert type_utils.match_annotation(Union[int],int)
    assert not type_utils.match_annotation(Union[int],str)
    assert type_utils.match_annotation(Union[int,str],int)
    assert type_utils.match_annotation(Union[int,str],Union[int,None])
    assert not type_utils.match_annotation(Union[int,str],DummyClass)
    assert type_utils.match_annotation(Union[int,DummyClass],DummyClass)
    assert not type_utils.match_annotation(Tuple[int,str],DummyClass)
    assert type_utils.match_annotation(Tuple[int,str],Tuple[int,str])
    assert not type_utils.match_annotation(Tuple[int,str],Tuple[str,int])
    assert type_utils.match_annotation(tuple,tuple)
    assert type_utils.match_annotation(list,list)
    assert not type_utils.match_annotation(tuple,list)
    assert type_utils.match_annotation(List[int], List[int])
    assert not type_utils.match_annotation(List[int], List[str])
    assert type_utils.match_annotation(torch.Tensor, Union[torch.Tensor,np.ndarray])
    assert not type_utils.match_annotation(torch.Tensor, Union[np.ndarray])