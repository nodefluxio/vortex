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

    from pathlib import Path
    from vortex.utils.parser.loader import Loader
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

def add_parser(parent_parser,subparsers = None):
    if subparsers is None:
        parser = parent_parser
    else:
        parser = subparsers.add_parser('hypopt',description=description)
    parser.add_argument('-c','--config', required=True, type=str, help='path to experiment config file')
    parser.add_argument('-o','--optconfig', required=True, type=str, help='path to hypopt config file')
    parser.add_argument("-w","--weights", help='path to selected weights (optional, will be inferred from `output_directory` and `experiment_name` field from config) if not specified, valid only for ValidationObjective, ignored otherwise')

if __name__=='__main__' :
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=description)
    add_parser(parser)
    args = parser.parse_args()
    main(args)
