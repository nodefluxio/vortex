import logging
import onnx
import inspect
import parse
from typing import Union, Dict, List, Callable, Any
from .base_ops import GraphOpsBase
from .embed_metadata import EmbedMetadata
from .helper import get_metadata_prop

logger = logging.getLogger(__name__)

class EmbedClassNamesMetadata(GraphOpsBase):
    field_name = "class_labels"
    # label format used for both formatting and parsing (reverse format)
    label_fmt = "{:d}:{}"
    def __init__(self, class_names: Union[List[str],Dict]):
        self.class_names = class_names
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, class_names: Union[List[str],Dict]):
        """Embed class_names information to model

        Args:
            model (onnx.ModelProto): model
            class_names (Union[List[str],Dict]): class_names, either list of string or mapping from string to int

        Raises:
            TypeError: class_names is not dict nor list

        Returns:
            onnx.ModelProto: model
        """
        if not isinstance(class_names, (dict,list)):
            raise TypeError("expects class_names to be dictionary or list")
        if isinstance(class_names, list):
            class_names = dict(enumerate(class_names))
        labels = []
        for key, value in class_names.items():
            if not isinstance(key,int):
                raise TypeError("expects key in class_names to be integer")
            if not isinstance(value,(int,str)):
                raise TypeError("expects value in class_names to be string or integer")
            label_fmt = cls.label_fmt
            labels.append(label_fmt.format(key,value))
        labels_formatter = lambda x: '{}'.format(",".join(x))
        model = EmbedMetadata.apply(model,cls.field_name,labels,labels_formatter)
        return model
    
    @classmethod
    def parse(cls, model: onnx.ModelProto) -> Dict[int,str]:
        """Extract `class_names` information from model

        Args:
            model (onnx.ModelProto): model

        Raises:
            ValueError: if model doesn't contains `class_labels`

        Returns:
            Dict[int,str]: a mapping from class label (int) to class name
        """
        class_labels = get_metadata_prop(model, cls.field_name)
        if class_labels is None:
            raise ValueError("model doesn't contains classs_labels")
        # class label is onnx protobuf type (StringStringEntryProto)
        class_labels = str(class_labels.value)
        class_names  = dict()
        class_labels = class_labels.split(',')
        for label in class_labels:
            key, value = parse.parse(cls.label_fmt,label)
            class_names[key] = value
        return class_names

    def run(self, model: onnx.ModelProto):
        return self.apply(model,**vars(self))