import logging
import onnx
import json
from typing import Union, Dict, List
from .base_ops import GraphOpsBase
from .embed_metadata import embed_metadata, parse_metadata
from .helper import make_output_format

logger = logging.getLogger(__name__)

class EmbedOutputFormatMetadata(GraphOpsBase):
    # add pefix 'output' for clarity and easier parsing
    prefix = "output"
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
        for name, fmt in output_format.items():
            if not isinstance(fmt, dict):
                raise TypeError("expect values from output_format to be dictionary")
            if not 'indices' in fmt:
                raise ValueError("output_format doesnt contains 'indices'")
            if not 'axis' in fmt:
                raise ValueError("output_format doesnt contains 'axis'")
            model = embed_metadata(model,f"{cls.prefix}.{name}",fmt)
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
        f = lambda x: (x.key.replace(f"{cls.prefix}.",""), json.loads(str(x.value)))
        output_format = dict(map(f, outputs))
        return output_format
    
    @classmethod
    def _deprecated_apply(cls, model : onnx.ModelProto, output_format : Dict[str,Union[List[int],int]]) -> onnx.ModelProto:
        """
        embed output_format to model as `Constants`
        """
        logger.info("embedding output_format to model")
        output_format_constants, output_format_value_info = make_output_format(output_format)
        for output_format_constant, value_info in zip(output_format_constants, output_format_value_info):
            model.graph.node.append(output_format_constant)
            model.graph.output.append(value_info)
        return model
    
    def _deprecated_run(self, model: onnx.ModelProto):
        return self._deprecated_apply(model,**vars(self))
    
    def run(self, model: onnx.ModelProto):
        """run fn, as required by base class

        Args:
            model (onnx.ModelProto): model to be transformed

        Returns:
            onnx.ModelProto: transformed model
        """
        return self.apply(model, **vars(self))

# alias
embed_output_format_metadata = EmbedOutputFormatMetadata.apply

# alias
parse_output_format_metadata = EmbedOutputFormatMetadata.parse