from vortex.development.utils.registry import Registry

BACKBONES = Registry("Backbones")

register_backbone = BACKBONES.register
remove_backbone = BACKBONES.pop
