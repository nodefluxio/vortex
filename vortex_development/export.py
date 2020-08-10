import argparse

from vortex.utils.parser import load_config
from vortex.core.pipelines import GraphExportPipeline

description = "export model to specific IR specified in config, output IR are "\
        "stored in the experiment directory based on `experiment_name` under `output_directory` config field, after successful export, you should be able to visualize the "\
        "network using [netron](https://lutzroeder.github.io/netron/)"

def main(args):

    # Parse config
    config = load_config(args.config)

    # Initialize graph exporter
    graph_exporter=GraphExportPipeline(config=config,weights=args.weights)
    result = graph_exporter.run(example_input=args.example_input)
    if not result.export_status:
        raise RuntimeError("One or more IR export failed")
    print("DONE!")

def add_parser(parent_parser,subparsers = None):
    if subparsers is None:
        parser = parent_parser
    else:
        parser = subparsers.add_parser('export',description=description)
    parser.add_argument('-c','--config', required=True, help='export experiment config file')
    parser.add_argument("-w","--weights", help="path to selected weights (optional, will be inferred from "\
        "`output_directory` and `experiment_name` field from config) if not specified")
    parser.add_argument('-i',"--example-input", help="path to example input for tracing (optional, may be necessary "\
        "for correct tracing, especially for detection model)")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=description)
    add_parser(parser)
    args = parser.parse_args()
    main(args)
