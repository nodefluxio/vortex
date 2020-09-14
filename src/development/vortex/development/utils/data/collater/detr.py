import torch

from vortex.development.networks.models.detection.detr import NestedTensor
from easydict import EasyDict

class DETRCollate:
    def __init__(self, dataformat: dict):
        dataformat = {k: {n: torch.tensor(v, dtype=torch.long) if n == 'indices' else v for n,v in val.items()}
            for k,val in dataformat.items() if val is not None}
        self.dataformat = EasyDict(dataformat)
        self.disable_image_auto_pad = True

    def __call__(self, batch):
        images, targets = list(zip(*batch))
        images = NestedTensor.from_batch_tensor(images)

        collated_targets = []
        for target in targets:
            out_target = {}
            if target.ndim != 2:
                raise RuntimeError("expects dimensionality of target is 2 got %s" % target.ndim)
            if not "class_label" in self.dataformat or self.dataformat.class_label is None:
                out_target['labels'] = torch.zeros(target.shape[0], dtype=torch.int64)
            else:
                out_target['labels'] = torch.index_select(
                    input=target,
                    dim=self.dataformat.class_label.axis,
                    index=self.dataformat.class_label.indices
                ).flatten().type(torch.int64)
            out_target['bbox'] = torch.index_select(
                input=target,
                dim=self.dataformat.bounding_box.axis,
                index=self.dataformat.bounding_box.indices
            ).float()
            out_target['bbox'][:, :2] += out_target['bbox'][:, 2:] / 2  ## x,y,w,h -> cx,cy,w,h
            collated_targets.append(out_target)
        return images, collated_targets


supported_collater = ['DETRCollate']

def create_collater(collater: str, *args, **kwargs):
    if not collater in supported_collater:
        raise RuntimeError("unsupported collater %s, supported : %s" % (
            collater, ','.join(supported_collater)))
    return DETRCollate(*args, **kwargs)
