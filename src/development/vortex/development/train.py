import argparse

from vortex.development.core.pipelines import TrainingPipeline
from vortex.development.utils.parser import load_config

description='Vortex training pipeline; will generate a Pytorch model file'

def main(args):
    log_metric = not args.no_log

    # Load configuration from experiment file
    config = load_config(args.config)

    ckpt_in_cfg = 'checkpoint' in config and config.checkpoint is not None
    if ckpt_in_cfg:
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

def add_parser(subparsers, parent_parser):
    TRAIN_HELP = "Train model from configuration file"
    usage = "\n  vortex train [options] <config>"
    parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        description=TRAIN_HELP,
        help=TRAIN_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    parser.add_argument(
        "config", 
        help="path to experiment config file"
    )

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        '--resume', 
        action='store_true', 
        help='resume training, getting the weight from `config.checkpoint`'
    )
    cmd_args_group.add_argument(
        "--no-log", 
        action='store_true', 
        help='disable experiment logging'
    )

    parser.set_defaults(func=main)
