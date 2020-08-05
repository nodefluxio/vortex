import torch
import numpy as np

from easydict import EasyDict

class SSDCollate:
    """
    collater for ssd
    """

    def __init__(self, dataformat: dict) :
        ## TODO : check all necessary fields
        self.dataformat = EasyDict(dataformat)
        # self.disable_image_auto_pad = True
    
    def __call__(self, batch) :
        imgs, targets = list(zip(*batch))
        # import pdb; pdb.set_trace()
        df = self.dataformat

        list_targets = []

        for batch_idx, target in enumerate(targets):
            if not len(target.shape) == 2:
                raise RuntimeError(
                    "expects dimensionality of target is 2 got %s" % len(target.shape))
            if self.dataformat.class_label is None:
                class_label_tensor = torch.from_numpy(
                    np.array([[0]]*target.shape[0])).float()
            else:
                class_label_tensor = torch.index_select(
                    input=target,
                    dim=df.class_label.axis,
                    index=torch.tensor(
                        df.class_label.indices, dtype=torch.long)
                )
            bounding_box_tensor = torch.index_select(
                input=target,
                dim=df.bounding_box.axis,
                index=torch.tensor(df.bounding_box.indices, dtype=torch.long)
            )
            bounding_box_tensor[:,2:] = bounding_box_tensor[:,:2] + bounding_box_tensor[:,2:]
            target = torch.cat(
                (bounding_box_tensor, class_label_tensor), axis=1)            
            list_targets.append(target)

        return (torch.stack(imgs, 0), list_targets)
    
supported_collater = [
    'SSDCollate'
]

def create_collater(collater: str, *args, **kwargs):
    if not collater in supported_collater:
        raise RuntimeError("unsupported collater %s, supported : %s" % (
            collater, ','.join(supported_collater)))
    return eval(collater)(*args, **kwargs)