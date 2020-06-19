import numpy as np
from typing import Dict, List, Union
from easydict import EasyDict
from collections import namedtuple,OrderedDict

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
    result_type = namedtuple('Result', [*output_fields])

    final_results = []
    for result in results:
        temp_result = {
            key : (
                np.take(
                    result, 
                    indices=output_format[key].indices, 
                    axis=output_format[key].axis
                ) if all(result.shape) else None 
            )
            for key in output_format.keys()
         }
        temp_result = result_type(**temp_result)
        final_results.append(temp_result)
    results = [result._asdict() for result in final_results]

    return results