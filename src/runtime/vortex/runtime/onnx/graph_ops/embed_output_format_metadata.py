import logging
import onnx
import parse
from typing import Union, Dict, List
from .base_ops import GraphOpsBase
from .embed_metadata import EmbedMetadata

logger = logging.getLogger(__name__)

class EmbedOutputFormatMetadata(GraphOpsBase):
    # indices and axis value format
    indices_fmt = "{}"
    axis_fmt    = "{:d}"
    # add pefix 'output' for clarity and easier parsing
    prefix = "output"
    # output name format, used for formatting and parsing
    indices_key = "output.{name}.indices"
    axis_key    = "output.{name}.axis"
    def __init__(self, output_format: Dict[str,Union[List[int],int]]):
        self.output_format = output_format
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, output_format: Dict[str,Union[List[int],int]]):
        """Format each 'output' in 'output_format' and embed it to existing model

        Args:
            model (onnx.ModelProto): model
            output_format (Dict[str,Union[List[int],int]]): output_format, must contains 'indices' and 'axis' entry

        Raises:
            TypeError: value type of output_format is not dictionary
            ValueError: each entry in output_format doesnt contains 'indices' and/or 'axis'

        Returns:
            onnx.ModelProto: model with output_format embedded
        """
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
        """Extract and parse output_format information from model

        Args:
            model (onnx.ModelProto): model

        Returns:
            Dict[str,Union[List[int],int]]: output_format
        """
        # metadata_keys = map(lambda x: x.key, model.metadata_props)
        # note that model.metadata_props is RepeatedField of StringStringEntryProto from onnx.onnx_pb
        # filter metadata props that contains prefix
        outputs = filter(lambda x: x.key.startswith(f'{cls.prefix}.'),model.metadata_props)
        axes = {} # collections of axis for each output name
        indices = {} # collections of indices for each output name
        # outputs may contains both 'indices' and 'axis', separate to each kind
        for output in outputs:
            axis_key = parse.parse(cls.axis_key,output.key)
            indices_key = parse.parse(cls.indices_key,output.key)
            if axis_key:
                axes[axis_key['name']] = output.value
            else:
                indices[indices_key['name']] = output.value
        def to_indices(x: str):
            """given formatted string, return list representing indices

            Args:
                x (str): indices formatted as string, e.g. "[1,2,3]"

            Returns:
                list: indices for output_format
            """
            to_int = lambda x: int(x)
            x = x.replace('[','').replace(']','')
            x = list(to_int(s) for s in x.split(','))
            return x
        # axis parsing is simply reverse format
        to_axis = lambda x: parse.parse(cls.axis_fmt,x)[0]
        # for each indices and axis, apply corresponding transformation `to_indices` and `to_axis`
        indices = map(lambda k, v: (k, to_indices(v)), indices.keys(), indices.values())
        axes    = map(lambda k, v: (k, to_axis(v)), axes.keys(), axes.values())
        # finally group by output name
        output_format = {}
        for (k_ind,ind), (k_ax, ax) in zip(indices,axes):
            assert k_ind == k_ax, "broken format"
            output_format[k_ind] = dict(indices=ind,axis=ax)
        return output_format
    
    def run(self, model: onnx.ModelProto):
        """run fn, as required by base class

        Args:
            model (onnx.ModelProto): model to be transformed

        Returns:
            onnx.ModelProto: transformed model
        """
        return self.apply(model, **vars(self))
