import argparse

from vortex.development.core.pipelines import IRPredictionPipeline
from vortex.runtime import model_runtime_map

description = 'Vortex IR model prediction pipeline; may receive multiple image(s) for batched prediction'


def main(args):
    model_path=args.model
    test_images=args.image
    output_directory=args.output_dir
    runtime=args.runtime

    kwargs = vars(args)
    for key in ['model','image','runtime','output_dir']:
        kwargs.pop(key)

    available_runtime = []
    for runtime_map in model_runtime_map.values():
        available_runtime.extend(list(runtime_map.keys()))
    available_runtime = set(available_runtime)
    if runtime not in list(available_runtime):
        raise RuntimeError('Runtime "{}" is not available, available runtime = {}'.format(runtime,list(available_runtime)))

    # Initialize Vortex IR Predictor
    vortex_ir_predictor=IRPredictionPipeline(model=model_path,runtime=runtime)

    # Make prediction
    results = vortex_ir_predictor.run(images = test_images,
                                  visualize = True,
                                  dump_visual = True,
                                  output_dir = output_directory,
                                  **kwargs)
    prediction = results.prediction
    visualization = results.visualization

    # Convert class index to class names,obtain results
    if 'class_label' in prediction[0]:
        class_names = [[vortex_ir_predictor.model.class_names[int(class_index)] for class_index in result['class_label']] for result in prediction]
    else:
        class_names = [["class_0" for _ in result['class_confidence']] for result in prediction]
    print('Prediction : {}'.format(prediction))
    print('Class Names : {}'.format(class_names))

def add_parser(subparsers, parent_parser):
    IR_PREDICT_HELP = "Run prediction on IR model"
    usage = "\n  vortex predict [options] <model> <image ...>"
    parser = subparsers.add_parser(
        "ir_runtime_predict",
        parents=[parent_parser],
        description=IR_PREDICT_HELP,
        help=IR_PREDICT_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    parser.add_argument('model', type=str, help='path to IR model')
    parser.add_argument(
        "image", 
        nargs='+', type=str, 
        help="image(s) path to be predicted, supports up to model's batch size"
    )

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        "-o", "--output-dir",
        metavar="DIR",
        default='.',
        help="directory to dump prediction result"
    )
    cmd_args_group.add_argument(
        "-r", "--runtime", 
        type=str, default='cpu', 
        help='runtime device/backend to use for prediction'
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
