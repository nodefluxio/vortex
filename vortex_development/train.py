import argparse

from vortex.core.pipelines import TrainingPipeline
from vortex.utils.parser import load_config

description='Vortex training pipeline; will generate a Pytorch model file'

def main(args):
    log_metric = not args.no_log

    # Load configuration from experiment file
    config = load_config(args.config)

    ckpt_in_cfg = 'checkpoint' in config
    if ckpt_in_cfg:
        if config.checkpoint is not None:
            config.checkpoint = config.checkpoint.rstrip(',')
    if args.resume and (not ckpt_in_cfg or (ckpt_in_cfg and config.checkpoint == None)):
        ## TODO: point to documentation for resume
        raise RuntimeError("You choose to resume but 'checkpoint' is not specified. "
            "Please specify 'checkpoint' option in your configuration file pointing "
            "to model path used for resume. see documentation.")

    # Override logger config in experiment file
    if not log_metric:
        config.logging = None

    # create training pipeline
    train_executor = TrainingPipeline(config=config, config_path=args.config, 
                                      hypopt=False, resume=args.resume)
    output = train_executor.run()

    epoch_losses = output.epoch_losses
    val_metrics = output.val_metrics
    learning_rates = output.learning_rates

    print("\nloss: {}\nmetrics: {}\nlearning rates: {}".format(epoch_losses, 
        val_metrics, learning_rates))

def add_parser(parent_parser, subparsers = None):
    if subparsers is None:
        parser = parent_parser
    else:
        parser = subparsers.add_parser('train', description=description)
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='path to experiment config file')
    parser.add_argument('--resume', action='store_true', 
                        help='vortex-saved model path for resume training')
    parser.add_argument("--no-log", action='store_true', 
                        help='disable logging, ignore experiment file config')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    add_parser(parser)
    args = parser.parse_args()
    main(args)
