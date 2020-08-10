import argparse
from typing import Union, List

from vortex.utils.parser import load_config
from vortex.core.pipelines import PytorchPredictionPipeline

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
    vortex_predictor=PytorchPredictionPipeline(config = config,
                                     weights = weights_file,
                                     device = device)
    
    # Make prediction
    results = vortex_predictor.run(images = test_images,
                               visualize = True,
                               dump_visual = True,
                               output_dir = output_dir,
                               **kwargs)

    prediction = results.prediction
    visualization = results.visualization
    
    # Convert class index to class names,obtain results

    try :
        class_names = [[vortex_predictor.model.class_names[int(class_index)] for class_index in result['class_label']] for result in prediction]
    except :
        class_names = None
    print('Prediction : {}'.format(prediction))
    print('Class Names : {}'.format(class_names))

def add_parser(parent_parser,subparsers = None):
    if subparsers is None:
        parser = parent_parser
    else:
        parser = subparsers.add_parser('predict',description=description)
    parser.add_argument("-c","--config", required=True, help='path to experiment config')
    parser.add_argument("-w","--weights", help='path to selected weights(optional, will be inferred from `output_directory` and `experiment_name` field from config) if not specified')
    parser.add_argument("-o","--output-dir",default='.',help='directory to dump prediction visualization')
    parser.add_argument("-i","--image", required=True, nargs='+', type=str, help='path to test image(s)')
    parser.add_argument('-d',"--device", help="the device in which the inference will be performed")

    # Additional arguments for detection model
    parser.add_argument("--score_threshold", default=0.9, type=float,
                        help='score threshold for detection, only used if model is detection, ignored otherwise')
    parser.add_argument("--iou_threshold", default=0.2, type=float,
                        help='iou threshold for nms, only used if model is detection, ignored otherwise')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    add_parser(parser)
    args = parser.parse_args()

    main(args)


