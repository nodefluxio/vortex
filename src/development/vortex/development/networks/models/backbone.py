import torch
import torch.nn as nn

from typing import Tuple, Union, Sequence

from vortex.development.networks.modules.backbones import get_backbone, BackboneBase


class ModuleIOHook:
    """Forward hook class for nn.Module to retrieve module's input and output.
    """
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.output = None
        self.input = None

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Backbone(nn.Sequential):

    _stages_output_str_map = {
        'tri_stage_fpn': [2, 3, 4],
        'tri_stage': [2, 3, 4],
        'classifier': [-1],
    }

    def __init__(
        self,
        module: Union[str, BackboneBase, nn.Sequential, nn.ModuleList, Sequence],
        stages_output: Union[str, int, Sequence] = [2, 3, 4],
        stages_channel: Union[Sequence] = None,
        freeze: bool = False,
        name: str = None,
        **kwargs
    ):
        super().__init__()

        stages = None
        reduce_stages_channel = False
        if isinstance(module, str):
            module = get_backbone(module, **kwargs)

        if isinstance(module, BackboneBase):
            name = module.name
            stages = module.get_stages()
            stages.add_module("classifier", module.get_classifier())
            if stages_channel is None:
                reduce_stages_channel = True
                stages_channel = list(module.stages_channel) + [module.num_classes]
        elif isinstance(module, (nn.Sequential, nn.ModuleList)):
            stages = module
        elif isinstance(module, Sequence):
            stages = nn.Sequential(*module)
        else:
            raise TypeError("'module' argument is expected to have 'str', 'BackboneBase', "
                "'nn.Sequential', 'nn.ModuleList', or list of 'nn.Module' type, got {}".format(type(module)))

        self.stages_output = self._validate_stages_output(stages_output, len(stages), self._stages_output_str_map)

        max_stages = max(self.stages_output)
        for idx, (n, module) in enumerate(stages.named_children()):
            if idx > max_stages:
                break
            self.add_module(n, module)

        assert len(self) == max_stages+1, "This should not happened, please report as bug."

        if freeze:
            self.requires_grad_(False)
        self.freeze = freeze

        self._name = name if name else "backbone"
        self.hooks_handle = [ModuleIOHook(m) for n,m in enumerate(self) if n in self.stages_output]

        if stages_channel is None:
            module_out = self.forward(torch.randn(2, 3, 224, 224))
            stages_channel = [o.size(1) for o in module_out]
        if reduce_stages_channel:
            stages_channel = [x for n,x in enumerate(stages_channel) if n in self.stages_output]

        if len(stages_channel) != len(self.stages_output):
            raise RuntimeError("length of 'stages_channel' ({}) and number of 'stages_output' ({}) "
                "is expected to be equal.".format(len(stages_channel), len(self.stages_output)))
        self._stages_channel = tuple(stages_channel)

    def forward(self, x) -> Tuple:
        x = super().forward(x)
        return tuple(h.output for h in self.hooks_handle)

    @property
    def stages_channel(self) -> Tuple:
        return self._stages_channel

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _validate_stages_output(stages_output, len_stages, stages_output_str_map):
        if isinstance(stages_output, str):
            if stages_output not in stages_output_str_map:
                raise ValueError("str type 'stages_output' with value of '{}' is not valid. "
                    "Supported values: {}".format(stages_output, list(stages_output_str_map)))
            stages_output = stages_output_str_map[stages_output]
        elif isinstance(stages_output, int):
            stages_output = [stages_output]

        if not isinstance(stages_output, Sequence):
            raise TypeError("'stages_output' is expected to have 'str', 'int' or any sequence type, "
                "got {}".format(type(stages_output)))

        stages_output_val = [x if x >= 0 else len_stages+x for x in stages_output]
        stages_output_val = tuple(set(stages_output_val))    ## ignore same value

        ## check stages validity
        if any((x >= len_stages or x < 0) for x in stages_output_val):
            raise RuntimeError("Invalid stages output value found, all value of 'stages_output' ({}) can't "
                "be higher than the total number of stages ({}).".format(stages_output, len_stages))
        return stages_output_val
