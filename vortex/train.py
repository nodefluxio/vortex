import argparse

from vortex.utils.parser import load_config, check_config
from vortex.core.pipelines import TrainingPipeline

description='Vortex training pipeline; will generate a Pytorch model file'

def main(args):
    config_path = args.config
    log_metric = not args.no_log

    # Load configuration from experiment file
    config = load_config(config_path)

    # Override logger config in experiment file
    if not log_metric:
        config.logging = 'None'

    # Instantiate Training
    train_executor = TrainingPipeline(config=config,config_path=config_path,hypopt=False)
    output = train_executor.run()

    epoch_losses = output.epoch_losses
    val_metrics = output.val_metrics
    learning_rates = output.learning_rates

    print(epoch_losses,val_metrics,learning_rates)

def add_parser(parent_parser,subparsers = None):
    if subparsers is None:
        parser = parent_parser
    else:
        parser = subparsers.add_parser('train',description=description)
    parser.add_argument('-c', '--config', required=True, help='path to experiment config file')
    parser.add_argument("--no-log", action='store_true', help='disable logging, ignore experiment file config')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    add_parser(parser)
    args = parser.parse_args()
    main(args)