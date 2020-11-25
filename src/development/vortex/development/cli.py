import sys
import argparse
import logging

logger = logging.getLogger("vortex")

from vortex.development.command import (
    train,
    validate,
    export,
    hypopt,
    predict,
    ir_runtime_predict,
    ir_runtime_validate
)
from vortex.development import __version__

COMMAND = [
    train,
    validate,
    export,
    predict,
    hypopt,
    ir_runtime_predict,
    ir_runtime_validate
]

class VersionAction(argparse.Action):
    """Shows version and exits."""

    def __call__(self, parser, namespace, values, option_string=None):
        print(__version__)
        sys.exit(0)


def get_main_parser():
    parent_parser = argparse.ArgumentParser(add_help=False)

    log_level_group = parent_parser.add_mutually_exclusive_group()
    log_level_group.add_argument(
        "-v", "--verbose", action="count", default=0, 
        help="Increase verbosity. Option is additive up to 3 times, e.g. `-vv`"
    )
    log_level_group.add_argument(
        "-q", "--quiet", action="count", default=0, 
        help="Be more quiet. Option is additive up to 2 times."
    )

    usage = "\n  vortex <COMMAND> [options]"
    parser = argparse.ArgumentParser(
        prog="vortex",
        description="Vortex command line tool",
        parents=[parent_parser],
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        usage=usage
    )

    parser.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )

    parser.add_argument(
        "-V", "--version",
        nargs=0,
        action=VersionAction,
        help="Show installed version."
    )

    ## sub commands
    subparsers = parser.add_subparsers(
        title="Available Commands",
        metavar="COMMAND",
        dest="cmd",
        help="Use `vortex COMMAND --help` for command-specific help."
    )
    subparsers.required = True
    subparsers.dest = "cmd"

    for cmd in COMMAND:
        cmd.add_parser(subparsers, parent_parser)

    return parser

def parse_args(argv):
    parser = get_main_parser()
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING
    if args.quiet == 1:
        level = logging.ERROR
    elif args.quiet >= 2:
        level = logging.CRITICAL
    elif args.verbose == 1:
        level = logging.INFO
    elif args.verbose == 2:
        level = logging.DEBUG
    elif args.verbose >= 3:
        level = logging.TRACE
    logging.basicConfig(level=level)

    ## run command function from `parser.set_defaults`
    args.func(args)

    ## TODO: return meaningful status
    return 0

if __name__ == "__main__":
    sys.exit(main())
