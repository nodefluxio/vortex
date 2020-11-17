import yaml
import logging
import argparse
from easydict import EasyDict

from vortex.development.core.pipelines import HypOptPipeline

logger = logging.getLogger(__name__)
description = "Vortex hyperparameter optimization experiment"


def main(args):
    config_path=args.config
    optconfig_path=args.optconfig
    weights=args.weights

    from vortex.development.utils.parser.loader import Loader
    with open(config_path) as f:
        config_data = yaml.load(f, Loader=Loader)
    with open(optconfig_path) as f:
        optconfig_data = yaml.load(f, Loader=Loader)

    config = EasyDict(config_data)
    optconfig = EasyDict(optconfig_data)
    hypopt = HypOptPipeline(config=config,optconfig=optconfig,weights=weights)
    result = hypopt.run()
    print(result.best_trial)

# Optional package for optuna visualization

## nodejs >= 6.0.0
## apt install nodejs
## apt install npm
## apt install gtk2.0
## apt -y install libgconf2-4
## apt install -y libnss3-tools
## apt install -y xvfb
## npm install -g electron@1.8.4 orca
## pip install plotly psutil requests cma

def add_parser(subparsers, parent_parser):
    HYPOPT_HELP = "Run hyperparameter optimization"
    usage = "\n  vortex hypopt [options] <config> <optconfig>"
    parser = subparsers.add_parser(
        "hypopt",
        parents=[parent_parser],
        description=HYPOPT_HELP,
        help=HYPOPT_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    parser.add_argument(
        'config', type=str, 
        help='path to experiment config file'
    )
    parser.add_argument(
        'optconfig', type=str, 
        help='path to hypopt config file'
    )

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        "-w", "--weights", 
        help="path to model's weights (optional, inferred from config if not specified)"
             "valid only for ValidationObjective, ignored otherwise"
    )

    parser.set_defaults(func=main)
