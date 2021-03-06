import argparse
import logging
import warnings

from vortex.development.utils.parser import load_config, check_config
from vortex.runtime import model_runtime_map
from vortex.development.core.pipelines import IRValidationPipeline

description = "Vortex exported IR graph validation pipeline; successful runs will produce autogenerated reports"

logger = logging.getLogger(__name__)


def check_deprecated_args(args):
    if args.config is None and args.config_dep is not None:
        warnings.warn("Argument `--config` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex ir_runtime_validate config.yml model.onnx`.", DeprecationWarning)
        args.config = args.config_dep
    elif args.config is not None and args.config_dep is not None:
        warnings.warn("Both positional and optional argument for config file "
            "is given, will use the positional argument instead.")
    elif args.config is None and args.config_dep is None:
        raise RuntimeError("config argument is not given, make sure to specify "
            "it, e.g. `$ vortex ir_runtime_validate config.yml model.onnx`")

    if args.model is None and args.model_dep is not None:
        warnings.warn("Argument `--model` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex ir_runtime_validate config.yml model.onnx`.", DeprecationWarning)
        args.model = args.model_dep
    elif args.model is not None and args.model_dep is not None:
        warnings.warn("Both positional and optional argument for IR model "
            "file is given, will use the positional argument instead.")
    elif args.model is None and args.model_dep is None:
        raise RuntimeError("model path argument is not given, make sure to specify "
            "it, e.g. `$ vortex ir_runtime_validate config.yml model.onnx`")


def main(args):
    check_deprecated_args(args)

    available_runtime = []
    for runtime_map in model_runtime_map.values():
        available_runtime.extend(list(runtime_map.keys()))
    available_runtime = set(available_runtime)
    for runtime in args.runtime:
        if runtime not in list(available_runtime):
            raise RuntimeError('Runtime "{}" is not available, available runtime = {}'.format(runtime,list(available_runtime)))

    # Parse config
    config = load_config(args.config)
    check_result = check_config(config, 'validate')
    logging.debug(check_result)
    if not check_result.valid:
        raise RuntimeError("invalid config : %s" % str(check_result))

    # Initialize IR validator

    validation_executor = IRValidationPipeline(config=config,
                                            model = args.model,
                                            backends = args.runtime,
                                            generate_report = True)
    eval_results = validation_executor.run(batch_size=args.batch_size)

    if 'pr_curves' in eval_results :
        eval_results.pop('pr_curves')
    print('validation results: {}'.format(
        ', '.join(['{}: {}'.format(key, value) for key, value in eval_results.items()])
    ))


def add_parser(subparsers, parent_parser):
    IR_VALIDATE_HELP = "run validation on IR model from configuration file"
    usage = "\n  vortex validate [options] <config> <model>"
    parser = subparsers.add_parser(
        "ir_runtime_validate",
        parents=[parent_parser],
        description=IR_VALIDATE_HELP,
        help=IR_VALIDATE_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    parser.add_argument(
        "config", nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to experiment config file."
    )
    parser.add_argument(
        "model",
        type=str, nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to IR model"
    )

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        "-r", "--runtime",
        nargs="*",
        type=str, default=['cpu'], 
        help='runtime device/backend, multiple values are possible'
    )
    cmd_args_group.add_argument(
        "-b", "--batch-size", 
        type=int, 
        help="batch size for validation, this value must match with exported model batch size"
    )

    deprecated_group = parser.add_argument_group(title="deprecated arguments")
    deprecated_group.add_argument(
        "-c", "--config",
        dest="config_dep", metavar="CONFIG",
        help="path to experiment config file.\nThis argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )
    deprecated_group.add_argument(
        "-m", "--model",
        dest="model_dep", metavar="MODEL",
        help="path to IR model.\nThis argument is DEPRECATED and will be "
             "removed. Use the positional argument instead."
    )

    parser.set_defaults(func=main)
