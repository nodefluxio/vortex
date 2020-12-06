import numpy as np
from typing import Dict, List, Union
from easydict import EasyDict
from collections import OrderedDict

__all__ = [
    'get_prediction_results'
]


def get_prediction_results(results : np.ndarray, output_format : Dict[str,Union[List[int],int]]) -> List[Dict[str,np.ndarray]]:
    if not isinstance(output_format, EasyDict):
        output_format = EasyDict(output_format)
    for k in output_format:
        if isinstance(output_format[k].indices, dict):
            indices = output_format[k].indices
            output_format[k].indices = [x for x in range(indices['start'], indices['end'])]

    output_fields = sorted(output_format.keys())

    final_results = []
    for result in results:
        temp_result = OrderedDict()
        for key in output_fields:
            temp_result[key] = np.take(
                                    result, 
                                    indices=output_format[key].indices, 
                                    axis=output_format[key].axis
                                ) if all(result.shape) else None 

        final_results.append(temp_result)

    return final_results