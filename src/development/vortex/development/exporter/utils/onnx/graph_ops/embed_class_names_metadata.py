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
    label_fmt = "{:d}:{}"
    def __init__(self, class_names: Union[List[str],Dict]):
        self.class_names = class_names
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, class_names: Union[List[str],Dict]):
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
        class_labels = get_metadata_prop(model, 'class_labels')
        if class_labels is None:
            raise ValueError("model doesn't contains classs_labels")
        # class label is protobuf type
        class_labels = str(class_labels.value)
        class_names  = dict()
        class_labels = class_labels.split(',')
        for label in class_labels:
            key, value = parse.parse(cls.label_fmt,label)
            class_names[key] = value
        return class_names

    def run(self, model: onnx.ModelProto):
        return self.apply(model,**vars(self))