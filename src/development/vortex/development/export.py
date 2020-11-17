import argparse

from vortex.development.utils.parser import load_config
from vortex.development.core.pipelines import GraphExportPipeline

description = "export model to specific IR specified in config, output IR are "\
        "stored in the experiment directory based on `experiment_name` under `output_directory` config field, after successful export, you should be able to visualize the "\
        "network using [netron](https://lutzroeder.github.io/netron/)"

def main(args):

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
        "config", 
        help="path to experiment config file"
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

    parser.set_defaults(func=main)
