from typing import Tuple,List,Union
import numpy as np

def _check_and_convert_limit_value(value,minimum = 0,modifier = 1):
    if isinstance(value,List) or isinstance(value,Tuple):
        if len(value)!=2 or value[0]>value[-1]:
            raise ValueError('Limit must be provided as list/tuple with length 2 -> [min,max] value, \
                                found {}'.format(value))
    elif isinstance(value,int) or isinstance(value,float):
        value = [-value,value]
        value = np.array(value) + modifier
    else:
        import pdb; pdb.set_trace()    
        raise ValueError('Unknown limit type, expected to be list/tuple or int, found {}'.format(value))
    
    if minimum:
        if value[0] < minimum:
            raise ValueError('Minimum value limit is 0, found {}'.format(value[0]))

    value = value.tolist()
    return value