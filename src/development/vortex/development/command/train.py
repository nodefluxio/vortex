import argparse
import warnings

from vortex.development.pipelines import TrainingPipeline
from vortex.development.utils.parser import load_config


def check_deprecated_args(args):
    if args.config is None and args.config_dep is not None:
        warnings.warn("Argument `--config` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex train config.yml`.", DeprecationWarning)
        args.config = args.config_dep
    elif args.config is not None and args.config_dep is not None:
        warnings.warn("Both positional and optional argument for config file "
            "is given, will use the positional argument instead.")
    elif args.config is None and args.config_dep is None:
        raise RuntimeError("config argument is not given, make sure to "
            "specify it, e.g. `$ vortex train config.yml`")


def main(args):
    check_deprecated_args(args)

    # create training pipeline
    train_executor = TrainingPipeline(args.config, hypopt=False, resume=args.resume)
    train_executor.run()


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
        "config", nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to experiment config file."
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

    deprecated_group = parser.add_argument_group(title="deprecated arguments")
    deprecated_group.add_argument(
        "-c", "--config",
        dest="config_dep", metavar="CONFIG",
        help="path to experiment config file.\nThis argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )

    parser.set_defaults(func=main)
