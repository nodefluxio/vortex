from vortex.core.factory import create_dataloader
from easydict import EasyDict
import numpy as np

if __name__ == "__main__":
    preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [0,0,0],
            'std' : [1, 1, 1],
            'scaler' : 255
        },
    })

    # Obj Detection
    # collate_fn = 'SSDCollate'
    # dataset_config = EasyDict(
    #     {
    #         'train': {
    #             'dataset' : 'VOC2007DetectionDataset',
    #             'args' : {
    #                 'image_set' : 'train'
    #             }
    #         }
    #     }
    # )

    # Classification
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

    # Obj Det with Landmark
    collate_fn = 'RetinaFaceCollate'
    dataset_config = EasyDict(
        {
            'train': {
                'dataset': 'FrontalFDDBDataset',
                'args': {
                    'train': True
                },
            }
        }
    )

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

    for datas in dataloader:
        dali_data = datas
        break
    
    # test vis
    import cv2
    
    image_index = 2
    vis = dali_data[0][image_index].cpu()
    vis = vis.mul(preprocess_args.input_normalization.scaler)

    vis = np.transpose(vis.numpy(), (1,2,0)).copy()
    # import pdb; pdb.set_trace()
    h,w,c = vis.shape

    allbboxes = dali_data[1][image_index][:,:4]

    for bbox in allbboxes:
        x = int(bbox[0]*w)
        y = int(bbox[1]*h)
        x2 = int(bbox[2]*w)
        y2 = int(bbox[3]*h)
        cv2.rectangle(vis, (x, y),(x2, y2), (0, 0, 255), 2)

    alllandmarks = dali_data[1][image_index][:,4:14]
    
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
            
    cv2.imshow('dali', vis.astype('uint8'))
    cv2.waitKey(0)

    # dataset = create_dataset(dataset_config, stage="train", preprocess_config=preprocess_args,wrapper_format='default')

    # dataloader_module_args = {
    #     'num_workers' : 0,
    #     'batch_size' : 4,
    #     'shuffle' : False,
    # }

    # dataloader = DataLoader(dataset, collate_fn=collate_fn, **dataloader_module_args)

    # for datas in dataloader:
    #     pytorch_data = datas
    #     break

    # py_vis = pytorch_data[0][0].cpu().numpy().copy()
    # py_vis = np.transpose(py_vis, (1,2,0)).copy()

    # h,w,c = py_vis.shape

    # allbboxes = pytorch_data[1][0][:,:4]
    # for bbox in allbboxes:
    #     x = int(bbox[0]*w)
    #     y = int(bbox[1]*h)
    #     x2 = int(bbox[2]*w)
    #     y2 = int(bbox[3]*h)

    #     cv2.rectangle(py_vis, (x, y),(x2, y2), (0, 0, 255), 2)

    # alllandmarks = pytorch_data[1][0][:,4:14]
    
    # for obj in alllandmarks:
    #     landmarks = obj.reshape(5,2)
    #     for i,point in enumerate(landmarks):
    #         x = int(point[0]*w)
    #         y = int(point[1]*h)

    #         if i == 0 or i == 3:
    #             color = (255,0,0)
    #         else:
    #             color = (0,0,255)

    #         cv2.circle(py_vis,(x, y), 2, color, -1)

    # cv2.imshow('pytorch', py_vis)
    # cv2.waitKey(0)
    # import pdb; pdb.set_trace()

