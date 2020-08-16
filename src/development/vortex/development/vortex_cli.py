import os
import sys
from pathlib import Path
import argparse
import logging
logger = logging.getLogger(__name__)

from vortex.development import (
    train,
    validate,
    export,
    hypopt,
    predict,
    ir_runtime_predict,
    ir_runtime_validate
)

STAGES = [
    train,
    validate,
    export,
    hypopt,
    predict,
    ir_runtime_predict,
    ir_runtime_validate
]

def main():
    parent_parser = argparse.ArgumentParser(description='Vortex CLI Development Tools')
    subparsers = parent_parser.add_subparsers(description='Vortex pipeline stage selection',dest='stage')
    subparsers.required = True
    subparsers.dest = "stage"
    for stage in STAGES:
        stage.add_parser(parent_parser, subparsers)
    args = parent_parser.parse_args()
    selected_stage = args.stage
    eval(selected_stage).main(args)

if __name__ == "__main__":
    main()