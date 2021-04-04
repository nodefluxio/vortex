import logging
import onnx

from .helper import make_class_names, get_outputs
from .base_ops import GraphOpsBase
from typing import Dict, List

logger = logging.getLogger(__name__)

class EmbedClassNames(GraphOpsBase):
    def __init__(self, class_names: Dict[str,int]):
        """Embed class_names to model as `Constants`

        Args:
            class_names (Dict[str,int]): mapping from string to int respresenting class_names
        """        
        self.class_names = class_names
    
    @classmethod
    def apply(cls, model : onnx.ModelProto, class_names : Dict[str,int]) -> onnx.ModelProto:
        """
        embed class_names to model as `Constants`
        """
        logger.info("embedding class_names to model")
        class_names_constants, class_names_value_info = make_class_names(class_names)
        for class_names_constant, value_info in zip(class_names_constants, class_names_value_info):
            model.graph.node.append(class_names_constant)
            model.graph.output.append(value_info)
        return model
    
    @classmethod
    def parse(cls, model : onnx.ModelProto, ignore_suffix=['_axis', '_indices', '_label']) -> List[str] :
        outputs = get_outputs(model)
        ignored = lambda x : any(suffix in x for suffix in ignore_suffix)
        output_names = [op.name for op in outputs if not ignored(op.name)]
        return output_names
    
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Actually embed class_names to model

        Args:
            model (onnx.ModelProto): model to be embedded with class_names

        Returns:
            onnx.ModelProto: transformed model
        """        
        return self.apply(model, **vars(self))