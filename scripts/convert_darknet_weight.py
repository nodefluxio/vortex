"""Darknet Weight Converter to Vortex Model

This is a utility script to convert darknet weight to vortex.
For normal usage, you need to have darknet model cfg, model weight, 
and (if you have one) the class names file.

For example, if you want to convert YOLOv3, use the cfg file from
https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg;
then the model's weight corresponding to the cfg file, from
https://pjreddie.com/media/files/yolov3.weights;
also you need the class names for the model, which is coco in this case
https://github.com/AlexeyAB/darknet/blob/master/data/coco.names;
Run the script using:
```
$ python convert_darknet_weight.py --model yolov3 --cfg yolov3.cfg
  --weight yolov3.weights --names coco.names
```

For more detailed arguments this script can accept, see with:
```
$ python convert_darknet_weight.py --help
```
"""

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


cfg_template_yolo = \
"""experiment_name: {exp_name}
device: 'cuda:0'
checkpoint: {weight_file}
output_directory: experiments/outputs
model: {{
  name: YoloV3,
  preprocess_args: {{
    input_size: {input_size},
    input_normalization: {{
      mean: [0.0, 0.0, 0.0],
      std: [1.0, 1.0, 1.0]
    }}
  }},
  network_args: {{
    backbone: darknet53,
    n_classes: {num_classes},
    anchors: {anchors},
    backbone_stages: {backbone_stages}
  }},
  loss_args: {{}},
  postprocess_args: {{
    nms: True,
    threshold: True,
  }}
}}
exporter: {{
  module: onnx,
  args: {{
    opset_version: 11,
  }}
}}
"""

cfg_template_darknet = \
"""experiment_name: {exp_name}
device: 'cuda:0'
checkpoint: {weight_file}
output_directory: experiments/outputs
model: {{
  name: softmax,
  preprocess_args: {{
    input_size: {input_size},
    input_normalization: {{
      mean: [0.0, 0.0, 0.0],
      std: [1.0, 1.0, 1.0]
    }}
  }},
  network_args: {{
    backbone: {model_name},
    n_classes: {num_classes}
  }},
  loss_args: {{}},
  postprocess_args: {{}}
}}
exporter: {{
  module: onnx,
  args: {{
    opset_version: 11,
  }}
}}
"""


msg = """Convert darknet weight to Vortex model definition
Notes:
- Implemented darknet models: yolov3, darknet7 (tiny-yolov3 backbone), darknet19 (yolov2 backbone), 
  darknet53 (yolov3 backbone)
- If you want to test the classification models, e.g. darknet53, use imagenet
  class names (for '--names' argument) defined in darknet:
  https://github.com/pjreddie/darknet/blob/master/data/imagenet.shortnames.list 
- If you want to get more accurate result, resize input image to rectangle (width == height) 
  and keep the original image aspect ratio by padding
"""

if __name__ == "__main__":
    print(msg)
    available_model = darknet.supported_models + ['yolov3']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--weight", required=True, type=str, help="darknet weight path")
    parser.add_argument("-c", "--cfg", required=True, type=str, help="file path for darknet config")
    parser.add_argument("--model", choices=available_model, default='yolov3',
        help="vortex model name to be converted (optional)")
    parser.add_argument("--names", type=str, 
        help="file path for darknet class names (optional)")
    parser.add_argument("-n", "--num-classes", type=int, 
        help="number of classes in model (optional)")
    parser.add_argument("--backbone-stages", nargs='+',
        help="list of backbone stages to output to head (optional)")
    parser.add_argument("--input-size", type=int, help="number of classes in model")
    parser.add_argument("--output", type=str, help="output file path (.pth)")
    args = parser.parse_args()

    weight_file = Path(args.weight)
    model_name = args.model
    if model_name is None:      ## automatically infer model name
        model_name = weight_file.name.split('.', 1)[0]
        if not model_name.lower().replace('-', '_') in available_model:
            raise RuntimeError("Unable to infer model name from darknet weight filename, "
                               "make sure to set the '--model' argument properly. \n"
                               "available name: {}".format(' '.join(available_model)))

    if args.output is None:
        args.output = Path(__file__).parent.joinpath("{}.pth".format(model_name))
    output_file = Path(args.output)
    if output_file.exists() and output_file.is_dir():
        fname = output_file.name
        output_file = output_file.joinpath(fname).with_suffix(".pth")
    if not output_file.name.endswith('.pth'):
        output_file = output_file.with_suffix('.pth')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ## cfg
    input_size = 256
    num_classes = 1000
    anchors = None
    cfg = None
    with open(args.cfg) as f:
        cfg = f.read().splitlines()

    ## input size
    if not args.input_size:
        wh_found = False
        for l in cfg:
            if ('height' in l or 'width' in l) and not l.startswith("#"):
                splt = l.strip().split('=')
                assert len(splt) == 2
                input_size = int(splt[-1])
                wh_found = True
                break
        if not wh_found:
            warnings.warn("Input size ('height' or 'width') is not found in your "
                "confif file, using default value of 256")
    else:
        input_size = args.input_size

    ## number of classes
    is_yolo = False
    if not args.num_classes:
        cfg_flipped = cfg[::-1].copy()
        ncls_found = False
        start, last_break = 0, 0
        for n,l in enumerate(cfg_flipped):
            if l.strip() == '[yolo]':
                start = n
                is_yolo = True
                break
            elif l.strip() == '[convolutional]':
                start = n
                break
            elif l.strip() == '' or l.strip().startswith('['):
                last_break = n
        if is_yolo and model_name != 'yolov3':
            model_name = 'yolov3'
        elif not is_yolo and model_name == 'yolov3':
            raise RuntimeError("Model name specified is 'yolov3', but YOLO Layer is not "
                "found in cfg file, make sure to properly specify '--model' argument")

        to_search = 'classes' if is_yolo else 'filters'
        block_cfg = cfg_flipped[start:last_break:-1].copy()
        anc_found = False
        for c in block_cfg:
            if to_search in c:
                splt = c.strip().split('=')
                assert len(splt) == 2
                num_classes = int(splt[-1])
                ncls_found = True
            if 'anchors' in c:
                splt = c.strip().split('=')
                assert len(splt) == 2
                anc_found = True
                anchors_tmp = [int(x) for x in splt[1].strip().split(',') if x.strip().isdigit()]
                assert len(anchors_tmp) == 18
                anchors = [(anchors_tmp[2*n], anchors_tmp[2*n+1]) for n in range(9)]
            if anc_found and ncls_found:
                break
        if not ncls_found:
            raise RuntimeError("number of class is not found in your config file, "
                "report this as a BUG!!")
    else:
        num_classes = args.num_classes

    ## backbone stages
    backbone_stages = None
    if is_yolo and not args.backbone_stages:
        cfg_flipped = cfg[::-1].copy()
        numlayer_to_stages = {4: 1, 11: 2, 36: 3, 61: 4, 74: 5}
        n_yolo_found, yolo_before = 0, True
        route_two_found = []
        for i,l in enumerate(cfg_flipped):
            if l.strip() == '[yolo]':
                yolo_before = True
                n_yolo_found += 1
                if n_yolo_found == 3:
                    break
            elif l.strip() == '[route]' and yolo_before:
                splt = cfg_flipped[i-1].strip().split('=')
                assert len(splt) == 2
                num_layers = splt[1].split(',')
                if len(num_layers) == 2:
                    route_two_found.append(int(num_layers[1].strip()))
                    yolo_before = False
        backbone_stages = [numlayer_to_stages[x] for x in route_two_found]
        if n_yolo_found == 3 and len(backbone_stages) == 2:
            backbone_stages.append(5)
        if len(backbone_stages) != 3:
            raise RuntimeError("Cannot found 3 route layers to determine backbone stages, "
                "report this as a bug!!")
    elif is_yolo:
        if len(args.backbone_stages) == 1 and isinstance(args.backbone_stages[0], str):
            args.backbone_stages = args.backbone_stages[0]
        backbone_stages = [int(x) for x in args.backbone_stages if x.isdigit()]

    print(">> Converting '{}' model with input size {} and number of classes {}"
        .format(model_name, input_size, num_classes))
    if model_name == "yolov3":
        model = YoloV3('darknet53', input_size, num_classes, anchors, backbone_stages=backbone_stages)
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

    ## generate vortex config
    if is_yolo:
        cfg_vortex = cfg_template_yolo.format(
            exp_name=output_file.name.replace('.pth', ''),
            weight_file=str(output_file),
            input_size=input_size,
            num_classes=num_classes,
            anchors=[list(x) for x in anchors],
            backbone_stages=backbone_stages
        )
    else:
        cfg_vortex = cfg_template_darknet.format(
            exp_name=output_file.name.replace('.pth', ''),
            weight_file=str(output_file),
            input_size=input_size,
            num_classes=num_classes,
            model_name=model_name
        )
    with open(output_file.with_suffix('.yml'), 'w+') as f:
        f.write(cfg_vortex)

    torch.save(checkpoint, output_file)
    print("DONE!!")
    print(">> model saved to: {}; with config in {}".format(output_file, output_file.with_suffix('.yml')))
