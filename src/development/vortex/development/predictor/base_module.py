import torch
import torch.nn as nn

from typing import Type, List, Union
from easydict import EasyDict

from vortex.development.networks.modules.postprocess.base_postprocess import NoOpPostProcess

class BasePredictor(nn.Module):
    """
    Base Predictor module used for inference and export.
    Required to have model, preprocess and postprocess;
    the following forward operation are defined :
    ```
    forward(input, **kwargs)
        x = preprocess(input)
        x = model(x)
        x = postprocess(x, **kwargs)
    ```
    the following signature are enforced :
    ```
        preprocess (input: Tensor) -> Tensor
        postprocess (input: Tensor, **kwargs) -> Tensor
    ```
    """
    postprocess_signature = {
        'input': torch.Tensor,
        # 'score_threshold': torch.Tensor,  # float,
        # 'iou_threshold': torch.Tensor,  # float,
        'return': Union[torch.Tensor,List[torch.Tensor]],
    }
    preprocess_signature = {
        'input': torch.Tensor,
        'return': torch.Tensor
    }

    def __init__(self, model: Type[nn.Module],
                 preprocess: Union[Type[nn.Module]] = None, 
                 postprocess: Union[Type[nn.Module]] = None):
        super(BasePredictor, self).__init__()

        self.model = model
        if preprocess is None:
            preprocess = nn.Identity()
        if postprocess is None:
            postprocess = NoOpPostProcess()
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.task = model.task

        if type(self.postprocess) != NoOpPostProcess:
            # force postprocess to be annotated
            if not len(self.postprocess.forward.__annotations__):
                raise RuntimeError("please annotate postprocess with %s" %
                                   BasePredictor.postprocess_signature.keys())
            annotations = self.postprocess.forward.__annotations__

            ## TODO : cleanup
            # for key, value in BasePredictor.postprocess_signature.items():
            #     if not key in annotations.keys():
            #         raise TypeError("missing postprocess annotation(s) : %s" % key)
            #     if value != annotations[key]:
            #         raise TypeError("type mismatch for `%s`, expects %s got %s" % (
            #             key, value, annotations[key]))

        if type(self.preprocess) != nn.modules.linear.Identity:
            # force preprocess to be annotated
            if not len(self.preprocess.forward.__annotations__):
                raise RuntimeError("please annotate postprocess with %s" %
                                   BasePredictor.preprocess_signature.keys())
            annotations = self.preprocess.forward.__annotations__
            for key, value in BasePredictor.preprocess_signature.items():
                if not key in annotations.keys():
                    raise TypeError("missing preprocess annotation(s) : %s" % key)
                if value != annotations[key]:
                    raise TypeError("type mismatch for `%s`, expects %s got %s" % (
                        key, value, annotations[key]))

        assert hasattr(self.model, 'output_format'), f"model definition '{type(model)}' "\
                "doesn't have 'output_format' attribute"
        self.output_format = EasyDict(self.model.output_format)

        required, optional = None, None
        if self.task == "detection":
            required = ["bounding_box", "class_confidence"]
            optional = ["landmarks", "class_label"]
        elif self.task == "classification":
            required = ["class_label", "class_confidence"]
        else:
            raise RuntimeError("unknown task of '%s'" % self.task)
        for elem in required:
            if not elem in self.output_format:
                raise RuntimeError("expects output_format for task %s to have `%s` key, "\
                    "got output_format: %s" % (self.task, elem, self.output_format))

        self.eval()

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.preprocess(input)
        x = self.model(x)
        x = self.postprocess(x, *args, **kwargs)
        return x


def create_predictor(model_components: Union[EasyDict, dict], **kwargs):
    if isinstance(model_components, dict):
        model_components = EasyDict(model_components)

    model = model_components.network
    preprocess = model_components.preprocess if 'preprocess' in model_components else None
    postprocess = model_components.postprocess if 'postprocess' in model_components else None

    return BasePredictor(model, preprocess=preprocess, postprocess=postprocess)
