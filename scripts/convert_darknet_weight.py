import sys
from pathlib import Path
project_path = Path(__file__).parents[2]
sys.path.append(str(project_path.joinpath('src', 'development')))

import torch
import argparse

from vortex.development.networks.models.detection.yolov3 import YoloV3
from vortex.development.networks.modules.backbones import darknet
from vortex.development.networks.modules.utils.darknet import load_darknet_weight


msg = """Convert darknet weight to Vortex model definition
Note:
- If you want to test the classification models, e.g. darknet53, use imagenet
  class names defined in darknet:
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
    parser.add_argument("--model-name", choices=available_model, 
        help="vortex model name to be converted (optional)")
    parser.add_argument("--names", type=str, help="file path for darknet class names (optional)")
    parser.add_argument("--output", type=str, help="output file path (.pth)")
    args = parser.parse_args()

    weight_file = Path(args.weight)
    model_name = args.model_name
    if model_name is None:      ## automatically infer model name
        model_name = weight_file.name.split('.', 1)[0]
        if not model_name.lower().replace('-', '_') in available_model:
            raise RuntimeError("Unable to infer model name from darknet weight filename, "\
                               "make sure to set the '--model-name' argument properly. \n"\
                               "available name: {}".format(' '.join(available_model)))
    output_file = args.output
    if output_file is None:
        output_file = project_path.joinpath("experiments", "outputs", "{}.pth".format(model_name))
        output_file = str(output_file)
    if not output_file.endswith('.pth'):
        output_file += '.pth'

    print("Converting '{}' model...".format(model_name))
    if model_name == "yolov3":
        model = YoloV3('darknet53', 608, 80)
    elif model_name in darknet.supported_models:
        model = getattr(darknet, model_name)(pretrained=False)
    else:
        raise RuntimeError("Unknown model name of {}".format(model_name))

    load_darknet_weight(model, weight_file)
    torch.save(model.state_dict(), output_file)
    print("DONE!!")
    print("model saved to: ", output_file)
