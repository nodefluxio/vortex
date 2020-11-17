import argparse
import torch

from vortex.development.utils.parser import load_config
from vortex.development.core.pipelines import PytorchPredictionPipeline

description = 'Vortex Pytorch model prediction pipeline; may receive multiple image(s) for batched prediction'

def main(args):
    config_path=args.config
    weights_file=args.weights
    test_images=args.image
    device=args.device
    output_dir=args.output_dir

    kwargs = vars(args)
    for key in ['config','weights','image','device','output_dir']:
        kwargs.pop(key)

    # Load experiment file
    config = load_config(config_path)
    
    # Initialize Vortex Vanila Predictor
    vortex_predictor = PytorchPredictionPipeline(config, weights=weights_file, device=device)
    
    # Make prediction
    results = vortex_predictor.run(images = test_images,
                               visualize = True,
                               dump_visual = True,
                               output_dir = output_dir,
                               **kwargs)

    prediction = results.prediction
    visualization = results.visualization

    # Convert class index to class names,obtain results
    if 'class_label' in prediction[0] and vortex_predictor.model.class_names is not None:
        class_names = [[vortex_predictor.model.class_names[int(class_index)] for class_index in result['class_label']] for result in prediction]
    else:
        class_names = [["class_0" for _ in result['class_confidence']] for result in prediction]
    print('Prediction : {}'.format(prediction))
    print('Class Names : {}'.format(class_names))

def add_parser(subparsers, parent_parser):
    PREDICT_HELP = "Run prediction on model from configuration file"
    usage = "\n  vortex predict [options] <config> <image ...>"
    parser = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        description=PREDICT_HELP,
        help=PREDICT_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    parser.add_argument(
        "config", 
        help="path to experiment config file"
    )
    parser.add_argument(
        "image", 
        nargs='+', type=str, 
        help="image(s) path to be predicted"
    )

    avail_devices = ["cpu"]
    if torch.cuda.is_available():
        num_device = torch.cuda.device_count()
        cuda_devices = ["cuda"] if num_device == 1 \
            else [f"cuda:{n}" for n in range(num_device)]
        avail_devices.extend(cuda_devices)

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        "-w", "--weights", 
        help="path to model's weights (optional, inferred from config if not specified)"
    )
    cmd_args_group.add_argument(
        "-o", "--output-dir",
        metavar="DIR",
        default='.',
        help="directory to dump prediction result"
    )
    cmd_args_group.add_argument(
        "-d", "--device",
        metavar="DEVICE",
        choices=avail_devices,
        help="the device in which the prediction is performed, "
             "available: {}".format(avail_devices)
    )

    # Additional arguments for detection model
    det_args_group = parser.add_argument_group(title="detection task arguments")
    det_args_group.add_argument(
        "--score_threshold", 
        default=0.9, type=float, 
        help="score threshold for detection nms"
    )
    det_args_group.add_argument(
        "--iou_threshold", 
        default=0.2, type=float, 
        help="iou threshold for detection nms"
    )

    parser.set_defaults(func=main)
