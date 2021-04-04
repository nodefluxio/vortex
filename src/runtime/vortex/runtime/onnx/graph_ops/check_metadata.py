from .embed_class_names_metadata import EmbedClassNamesMetadata
from .embed_output_format_metadata import EmbedOutputFormatMetadata
from .helper import get_output_names, get_output_format, get_input_specs

import onnx

def check_metadata(model: onnx.ModelProto) -> bool:
    # simple check for now
    try:
        class_names = EmbedClassNamesMetadata.parse(model)
    except:
        return False
    try:
        output_format = EmbedOutputFormatMetadata.parse(model)
    except:
        return False
    return True
    