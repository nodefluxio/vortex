from vortex.core.factory import create_dataloader
from easydict import EasyDict
import numpy as np

if __name__ == "__main__":
    preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [.3,.1,.2],
            'std' : [.4, .1, .2],
            'scaler' : 255
        },
    })

    # Obj Detection
    collate_fn = 'SSDCollate'
    dataset_config = EasyDict(
        {
            'train': {
                'dataset' : 'VOC2007DetectionDataset',
                'args' : {
                    'image_set' : 'train'
                }
            }
        }
    )

    # # Classification
    # collate_fn = None
    # dataset_config = EasyDict(
    #     {
    #         'train': {
    #             'dataset': 'ImageFolder',
    #             'args': {
    #                 'root': 'tests/test_dataset/train'
    #             },
    #         }
    #     }
    # )

    # # Obj Det with Landmark
    # collate_fn = 'RetinaFaceCollate'
    # dataset_config = EasyDict(
    #     {
    #         'train': {
    #             'dataset': 'FrontalFDDBDataset',
    #             'args': {
    #                 'train': True
    #             },
    #         }
    #     }
    # )

    dali_loader = EasyDict({
        'dataloader': 'DALIDataLoader',
        'args': {
            'device_id' : 0,
            'num_thread': 1,
            'batch_size': 4,
            'shuffle': False,
            },
    })

    pytorch_loader = EasyDict({
        'dataloader': 'PytorchDataLoader',
        'args': {
            'batch_size': 4,
            'shuffle': False,
        },
    })

    dataset_config.dataloader = dali_loader
    dataloader = create_dataloader(dataset_config=dataset_config,
                                   preprocess_config = preprocess_args,
                                   collate_fn=collate_fn)

    import cv2
    import torch

    for datas in dataloader:
        for i in range(datas[0].shape[0]):
            vis = datas[0][i].cpu()

            mean=torch.as_tensor(preprocess_args.input_normalization.mean, dtype=torch.float)
            std=torch.as_tensor(preprocess_args.input_normalization.std, dtype=torch.float)

            vis.mul_(std[:,None,None]).add_(mean[:,None,None])


            vis = vis.mul(preprocess_args.input_normalization.scaler)

            vis = np.transpose(vis.numpy(), (1,2,0)).copy()
            # import pdb; pdb.set_trace()
            h,w,c = vis.shape
        
            if 'bounding_box' in dataloader.dataset.data_format:

                allbboxes = datas[1][i][:,:4]

                for bbox in allbboxes:
                    x = int(bbox[0]*w)
                    y = int(bbox[1]*h)
                    x2 = int(bbox[2]*w)
                    y2 = int(bbox[3]*h)
                    cv2.rectangle(vis, (x, y),(x2, y2), (0, 0, 255), 2)

            if 'landmarks' in dataloader.dataset.data_format:
                alllandmarks = datas[1][i][:,4:14]
                
                for obj in alllandmarks:
                    landmarks = obj.reshape(5,2)
                    for i,point in enumerate(landmarks):
                        x = int(point[0]*w)
                        y = int(point[1]*h)

                        if i == 0 or i == 3:
                            color = (255,0,0)
                        else:
                            color = (0,0,255)

                        cv2.circle(vis,(x, y), 2, color, -1)

            cv2.imshow('vis', vis.astype('uint8'))
            cv2.waitKey(0)
