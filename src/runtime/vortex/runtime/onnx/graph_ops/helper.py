import onnx

def get_metadata_prop(model: onnx.ModelProto, key: str) -> onnx.onnx_pb.StringStringEntryProto:
    for prop in model.metadata_props:
        if prop.key == key:
            return prop
    return None