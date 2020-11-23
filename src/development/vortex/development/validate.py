import argparse
import logging
import warnings
import torch

from vortex.development.utils.parser import load_config, check_config
from vortex.development.core.pipelines import PytorchValidationPipeline

description='Vortex Pytorch model validation pipeline; successful runs will produce autogenerated reports'

logger = logging.getLogger(__name__)


def check_deprecated_args(args):
    if args.config is None and args.config_dep is not None:
        warnings.warn("Argument `--config` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex validate config.yml`.")
        args.config = args.config_dep
    elif args.config is not None and args.config_dep is not None:
        warnings.warn("Both positional and optional argument for config file "
            "is given, will use the positional argument instead.")
    elif args.config is None and args.config_dep is None:
        raise RuntimeError("config argument is not given, make sure to "
            "specify it, e.g. `$ vortex validate config.yml`")


def main(args):
    check_deprecated_args(args)

    config = load_config(args.config)
    check_result = check_config(config, 'validate')
    logger.debug(check_result)
    if not check_result.valid:
        raise RuntimeError("invalid config : %s" % str(check_result))
    weights_file=args.weights

    validation_executor = PytorchValidationPipeline(config=config,
                                                     weights = weights_file,
                                                     backends = args.devices,
                                                     generate_report = True)
    eval_results = validation_executor.run(batch_size=args.batch_size)
    if 'pr_curves' in eval_results :
        eval_results.pop('pr_curves')
    print('validation results: {}'.format(
        ', '.join(['{}: {}'.format(key, value) for key, value in eval_results.items()])
    ))


def add_parser(subparsers, parent_parser):
    VAL_HELP = "Validate model from configuration file and generate reports"
    usage = "\n  vortex validate [options] <config>"
    parser = subparsers.add_parser(
        "validate",
        parents=[parent_parser],
        description=VAL_HELP,
        help=VAL_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    parser.add_argument(
        "config", nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to experiment config file."
    )

    avail_devices = ["cpu"]
    if torch.cuda.is_available():
        num_device = torch.cuda.device_count()
        cuda_devices = ["cuda"] if num_device == 1 \
            else [f"cuda:{n}" for n in range(num_device)]
        avail_devices.extend(cuda_devices)

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        "-w","--weights", 
        help="path to model's weights (optional, inferred from config if not specified)"
    )
    cmd_args_group.add_argument(
        "-d", "--device",
        metavar="DEVICE",
        default=[],
        choices=avail_devices,
        help="device(s) in which the validation is performed; multiple values are "
             "possible, if not specified use device from config. available: {}"
             .format(avail_devices)
    )
    cmd_args_group.add_argument(
        "-b", "--batch-size", 
        default=1, type=int, 
        help="batch size for validation"
    )

    deprecated_group = parser.add_argument_group(title="deprecated arguments")
    deprecated_group.add_argument(
        "-c", "--config",
        dest="config_dep", metavar="CONFIG",
        help="path to experiment config file. This argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )

    parser.set_defaults(func=main)
