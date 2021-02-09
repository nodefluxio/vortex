import logging
import onnx
import parse
from typing import Union, Dict, List
from .base_ops import GraphOpsBase
from .embed_metadata import EmbedMetadata

logger = logging.getLogger(__name__)

class EmbedOutputFormatMetadata(GraphOpsBase):
    indices_fmt = "{}"
    axis_fmt    = "{:d}"
    # add pefix 'output' for clarity and easier parsing
    prefix = "output"
    indices_key = "output.{name}.indices"
    axis_key    = "output.{name}.axis"
    def __init__(self, output_format: Dict[str,Union[List[int],int]]):
        self.output_format = output_format
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, output_format: Dict[str,Union[List[int],int]]):
        # express list in indices_fmt
        indices_formatter = lambda x: cls.indices_fmt.format(str(x))
        axis_formatter = lambda x: cls.axis_fmt.format(x)
        for name, fmt in output_format.items():
            if not isinstance(fmt, dict):
                raise TypeError("expect values from output_format to be dictionary")
            if not 'indices' in fmt:
                raise ValueError("output_format doesnt contains 'indices'")
            if not 'axis' in fmt:
                raise ValueError("output_format doesnt contains 'axis'")
            indices = fmt['indices']
            axis    = fmt['axis']
            indices_key = cls.indices_key.format_map({'name': name})
            axis_key    = cls.axis_key.format_map({'name': name})
            # TODO: consider to add prefix, e.g. "output" so output.class_label.indices
            model = EmbedMetadata.apply(model,indices_key,indices,indices_formatter)
            model = EmbedMetadata.apply(model,axis_key,axis,axis_formatter)
        return model
    
    @classmethod
    def parse(cls, model: onnx.ModelProto) -> Dict[str,Union[List[int],int]]:
        # metadata_keys = map(lambda x: x.key, model.metadata_props)
        output_keys = filter(lambda x: x.key.startswith(f'{cls.prefix}.'),model.metadata_props)
        axes = {} # collections of axis for each output name
        indices = {} # collections of indices for each output name
        for output_key in output_keys:
            axis_key = parse.parse(cls.axis_key,output_key.key)
            indices_key = parse.parse(cls.indices_key,output_key.key)
            if axis_key:
                axes[axis_key['name']] = output_key.value
            else:
                indices[indices_key['name']] = output_key.value
        def to_indices(x: str):
            to_int = lambda x: int(x)
            x = x.replace('[','').replace(']','')
            x = list(to_int(s) for s in x.split(','))
            return x
        to_axis = lambda x: parse.parse(cls.axis_fmt,x)[0]
        indices = map(lambda k, v: (k, to_indices(v)), indices.keys(), indices.values())
        axes    = map(lambda k, v: (k, to_axis(v)), axes.keys(), axes.values())
        output_format = {}
        for (k_ind,ind), (k_ax, ax) in zip(indices,axes):
            assert k_ind == k_ax, "broken format"
            output_format[k_ind] = dict(indices=ind,axis=ax)
        return output_format
    
    def run(self, model: onnx.ModelProto):
        return self.apply(model, **vars(self))
