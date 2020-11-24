import argparse
import warnings
import torch

from vortex.development.utils.parser import load_config
from vortex.development.core.pipelines import PytorchPredictionPipeline

description = 'Vortex Pytorch model prediction pipeline; may receive multiple image(s) for batched prediction'


def check_deprecated_args(args):
    ## check config
    if args.config is None and args.config_dep is not None:
        warnings.warn("Argument `--config` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex predict config.yml image.jpg`.", DeprecationWarning)
        args.config = args.config_dep
    elif args.config is not None and args.config_dep is not None:
        warnings.warn("Both positional and optional argument for config file "
            "is given, will use the positional argument instead.")
    elif args.config is None and args.config_dep is None:
        raise RuntimeError("config argument is not given, make sure to "
            "specify it, e.g. `$ vortex predict config.yml image.jpg`")

    ## check image
    if args.image == [] and args.image_dep != []:
        warnings.warn("Argument `--image` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex predict config.yml image.jpg`.", DeprecationWarning)
        args.image = args.image_dep
    elif args.image != [] and args.image_dep != []:
        warnings.warn("Both positional and optional argument for image file(s) "
            "is given, will use the positional argument instead.")
    elif args.image == [] and args.image_dep == []:
        raise RuntimeError("image argument is not given, make sure to "
            "specify it, e.g. `$ vortex predict config.yml image.jpg`")


def main(args):
    check_deprecated_args(args)

    config_path = args.config
    weights_file = args.weights
    test_images = args.image
    device = args.device
    output_dir = args.output_dir

    kwargs = {
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold
    }

    # Load experiment file
    config = load_config(config_path)
    
    # Initialize Vortex Vanila Predictor
    vortex_predictor = PytorchPredictionPipeline(config, weights=weights_file, device=device)

    # Make prediction
    results = vortex_predictor.run(images = test_images,
                               visualize = (not args.no_visualize),
                               dump_visual = (not args.no_save),
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
        "config", nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to experiment config file"
    )
    parser.add_argument(
        "image", 
        nargs='*', type=str,    ## TODO: change `nargs` to '+' when deprecation is removed
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
    cmd_args_group.add_argument(
        "--no-visualize", 
        action='store_true', 
        help='not visualizing prediction result'
    )
    cmd_args_group.add_argument(
        "--no-save", 
        action='store_true', 
        help='not saving prediction result'
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

    deprecated_group = parser.add_argument_group(title="deprecated arguments")
    deprecated_group.add_argument(
        "-c", "--config",
        dest="config_dep", metavar="CONFIG",
        help="path to experiment config file. This argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )
    deprecated_group.add_argument(
        "-i", "--image",
        dest="image_dep", metavar="IMAGES",
        default=[],
        nargs='*', type=str, 
        help="image(s) path to be predicted. This argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )

    parser.set_defaults(func=main)
