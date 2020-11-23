import argparse
import warnings

from vortex.development.utils.parser import load_config
from vortex.development.core.pipelines import GraphExportPipeline

description = "export model to specific IR specified in config, output IR are "\
        "stored in the experiment directory based on `experiment_name` under `output_directory` config field, after successful export, you should be able to visualize the "\
        "network using [netron](https://lutzroeder.github.io/netron/)"


def check_deprecated_args(args):
    if args.config is None and args.config_dep is not None:
        warnings.warn("Argument `--config` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex export config.yml`.")
        args.config = args.config_dep
    elif args.config is not None and args.config_dep is not None:
        warnings.warn("Both positional and optional argument for config file "
            "is given, will use the positional argument instead.")
    elif args.config is None and args.config_dep is None:
        raise RuntimeError("config argument is not given, make sure to "
            "specify it, e.g. `$ vortex export config.yml`")


def main(args):
    check_deprecated_args(args)

    # Parse config
    config = load_config(args.config)
    if args.weights is None and hasattr(config, 'checkpoint') and config.checkpoint is not None:
        args.weights = config.checkpoint

    # Initialize graph exporter
    graph_exporter = GraphExportPipeline(config=config, weights=args.weights)
    result = graph_exporter.run(example_input=args.example_input)
    if not result.export_status:
        raise RuntimeError("One or more IR export failed")
    print("DONE!")


def add_parser(subparsers, parent_parser):
    EXPORT_HELP = "Export trained vortex model to IR model specified in configuration file"
    usage = "\n  vortex export [options] <config>"
    parser = subparsers.add_parser(
        "export",
        parents=[parent_parser],
        description=EXPORT_HELP,
        help=EXPORT_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    parser.add_argument(
        "config", nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to experiment config file."
    )

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        "-w", "--weights", 
        help="path to model's weights (optional, inferred from config if not specified)"
    )
    cmd_args_group.add_argument(
        "-i", "--example-input",
        metavar="IMAGE",
        help="path to example input for exporter tracing (optional, may be necessary "
             "to correctly trace the entire model especially for detection task)"
    )

    deprecated_group = parser.add_argument_group(title="deprecated arguments")
    deprecated_group.add_argument(
        "-c", "--config",
        dest="config_dep", metavar="CONFIG",
        help="path to experiment config file. This argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )

    parser.set_defaults(func=main)
