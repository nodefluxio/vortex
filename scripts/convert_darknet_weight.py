import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].joinpath('src', 'development')))

import torch
import argparse
import warnings

warnings.filterwarnings("ignore")
from vortex.development.networks.models.detection.yolov3 import YoloV3
from vortex.development.networks.modules.backbones import darknet
from vortex.development.networks.modules.utils.darknet import load_darknet_weight
warnings.resetwarnings()


msg = """Convert darknet weight to Vortex model definition
Notes:
- Implemented darknet models: yolov3, darknet7 (tiny-yolov3 backbone), darknet19 (yolov2 backbone), 
  darknet53 (yolov3 backbone)
- If you want to test the classification models, e.g. darknet53, use imagenet
  class names (for '--names' argument) defined in darknet:
  https://github.com/pjreddie/darknet/blob/master/data/imagenet.shortnames.list 
- Darknet models expect image input with of float with range [0,1] with RGB channels 
- If you want to get more accurate result, resize input image to rectangle (width == height) 
  and keep the original image aspect ratio by padding
"""

if __name__ == "__main__":
    print(msg)
    available_model = darknet.supported_models + ['yolov3']
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", required=True, type=str, help="darknet weight path")
    parser.add_argument("-c", "--cfg", required=True, type=str, help="file path for darknet config")
    parser.add_argument("--model", choices=available_model, default='yolov3',
        help="vortex model name to be converted (optional)")
    parser.add_argument("--names", type=str, help="file path for darknet class names (optional)")
    parser.add_argument("-n", "--num-classes", type=int, help="number of classes in model")
    parser.add_argument("--output", type=str, help="output file path (.pth)")
    args = parser.parse_args()

    weight_file = Path(args.weight)
    model_name = args.model
    if model_name is None:      ## automatically infer model name
        model_name = weight_file.name.split('.', 1)[0]
        if not model_name.lower().replace('-', '_') in available_model:
            raise RuntimeError("Unable to infer model name from darknet weight filename, "\
                               "make sure to set the '--model' argument properly. \n"\
                               "available name: {}".format(' '.join(available_model)))
    output_file = args.output
    if output_file is None:
        output_file = str(Path(__file__).parent.joinpath("{}.pth".format(model_name)))
    if not output_file.endswith('.pth'):
        output_file += '.pth'
    file_dir = Path(output_file).parent
    file_dir.mkdir(parents=True, exist_ok=True)

    ## cfg
    input_size = 256
    num_classes = 1000
    cfg = None
    with open(args.cfg) as f:
        cfg = f.read().splitlines()

    ## input size
    for l in cfg:
        if 'height' in l or 'width' in l:
            splt = l.strip().split('=')
            assert len(splt) == 2
            input_size = int(splt[-1])
            break

    ## number of classes
    if not args.num_classes:
        cfg_flipped = cfg[::-1].copy()
        is_yolo = False
        start, last_break = 0, 0
        for n,l in enumerate(cfg_flipped):
            if l.strip() == '':
                last_break = n
            elif l.strip() == '[yolo]':
                start = n
                is_yolo = True
                break
            elif l.strip() == '[convolutional]':
                start = n
                break

        to_search = 'classes' if is_yolo else 'filters'
        block_cfg = cfg_flipped[start:last_break:-1].copy()
        for c in block_cfg:
            if to_search in c:
                splt = c.strip().split('=')
                assert len(splt) == 2
                num_classes = int(splt[-1])
                break
        if is_yolo and not 'yolo' in model_name:
            raise RuntimeError("YOLO Layer is found in cfg file but your "
                "model name is not yolo variant model, got '{}'".format(model_name))
    else:
        num_classes = args.num_classes

    print("Converting '{}' model with input size {} and number of classes {}"
        .format(model_name, input_size, num_classes))
    if model_name == "yolov3":
        model = YoloV3('darknet53', input_size, num_classes)
    elif model_name in darknet.supported_models:
        model = getattr(darknet, model_name)(pretrained=False, num_classes=num_classes)
    else:
        raise RuntimeError("Unknown model name of {}".format(model_name))

    load_darknet_weight(model, weight_file)
    checkpoint = {'state_dict': model.state_dict()}

    if args.names:
        with open(args.names) as f:
            class_names = f.read().splitlines()
        assert len(class_names) >= num_classes, "Number of class names defined in {} " \
            "is less than number of class ({})".format(args.names, num_classes)
        checkpoint["class_names"] = class_names[:num_classes]

    torch.save(checkpoint, output_file)
    print("DONE!!")
    print("model saved to: ", output_file)
