import yaml
import warnings
import argparse
from easydict import EasyDict

from vortex.development.core.pipelines import HypOptPipeline

description = "Vortex hyperparameter optimization experiment"


def check_deprecated_args(args):
    if args.config is None and args.config_dep is not None:
        warnings.warn("Argument `--config` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex hypopt config.yml optconfig.yml`.", DeprecationWarning)
        args.config = args.config_dep
    elif args.config is not None and args.config_dep is not None:
        warnings.warn("Both positional and optional argument for config file "
            "is given, will use the positional argument instead.")
    elif args.config is None and args.config_dep is None:
        raise RuntimeError("config argument is not given, make sure to "
            "specify it, e.g. `$ vortex hypopt config.yml optconfig.yml`")

    if args.optconfig is None and args.optconfig_dep is not None:
        warnings.warn("Argument `--optconfig` is DEPRECATED and will be removed "
            "in the future. Use positional argument instead, e.g. "
            "`$ vortex hypopt config.yml optconfig.yml`.", DeprecationWarning)
        args.optconfig = args.optconfig_dep
    elif args.optconfig is not None and args.optconfig_dep is not None:
        warnings.warn("Both positional and optional argument for hypopt config "
            "file is given, will use the positional argument instead.")
    elif args.optconfig is None and args.optconfig_dep is None:
        raise RuntimeError("optconfig argument is not given, make sure to "
            "specify it, e.g. `$ vortex hypopt config.yml optconfig.yml`")


def main(args):
    check_deprecated_args(args)

    config_path = args.config
    optconfig_path = args.optconfig
    weights = args.weights

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
        'config',
        type=str, nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to experiment config file"
    )
    parser.add_argument(
        'optconfig',
        type=str, nargs="?",    ## TODO: remove `nargs` when deprecation is removed
        help="path to hypopt config file"
    )

    cmd_args_group = parser.add_argument_group(title="command arguments")
    cmd_args_group.add_argument(
        "-w", "--weights", 
        help="path to model's weights. Valid only for 'ValidationObjective', ignored otherwise."
             "\n(optional, inferred from config if not specified)"
    )

    deprecated_group = parser.add_argument_group(title="deprecated arguments")
    deprecated_group.add_argument(
        "-c", "--config",
        dest="config_dep", metavar="CONFIG",
        help="path to experiment config file.\nThis argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )
    deprecated_group.add_argument(
        "-o", "--optconfig",
        dest="optconfig_dep", metavar="OPTCONFIG",
        help="path to hypopt config file.\nThis argument is DEPRECATED "
             "and will be removed. Use the positional argument instead."
    )

    parser.set_defaults(func=main)
