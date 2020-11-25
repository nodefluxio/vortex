import sys
import argparse
import fnmatch
import blessed

from vortex.development.networks.modules import backbones
from vortex.development.networks import models


class ListBase:
    def __init__(self):
        self.term = blessed.Terminal()

    @classmethod
    def add_parser(cls, subparsers, parent_parser):
        raise NotImplementedError

    def _format_output(self, to_print: dict, header: str, fill_neg: int = 9, space: int = 8) -> str:
        max_char = len(max([max(val, key=lambda v: len(v)) for val in to_print.values()], key=lambda v: len(v)))
        fill_len = (max_char-fill_neg) if max_char > fill_neg else 1

        formatted = self.term.underline_bold(header.format(fill=" "*fill_len))
        for idx, (name, values) in enumerate(to_print.items()):
            if not values:
                continue
            if idx > 0:
                formatted += "\n"
            formatted += "{}:\n{}".format(name, " "*space)
            formatted += "\n{}".format(" "*space).join(v for v in values)
            formatted += "\n"
        return formatted


class BackboneList(ListBase):
    available_backbone = {
        n.__name__.split('.')[-1]: v for n,v in backbones.supported_models.copy().items()
    }

    def __init__(self):
        super().__init__()

    @classmethod
    def add_parser(cls, subparsers, parent_parser):
        HELP = "List available backbones"
        usage = "\n  vortex list backbone [options]"
        parser = subparsers.add_parser(
            "backbone",
            description=HELP,
            help=HELP,
            formatter_class=argparse.RawTextHelpFormatter,
            usage=usage
        )

        avail_print = ", ".join(list(cls.available_backbone.keys()))

        cmd_args_group = parser.add_argument_group(title="command arguments")
        cmd_args_group.add_argument(
            "-f", "--filter",
            type=str,
            help="Filter listed output using wildcard, e.g. '*resne*t*'"
        )
        cmd_args_group.add_argument(
            "--family",
            type=str, nargs="*",
            help="Only list specific backbone family, support multiple values.\n"
                "Available: {}".format(avail_print)
        )

        parser.set_defaults(cmp_cls=cls)

    def run(self, args):
        bb_to_print = dict()
        if args.family:
            bb_to_print.update(self._get_family(args.family))
        else:
            bb_to_print.update(self.available_backbone.copy())

        if args.filter:
            bb_to_print = self._filter(bb_to_print, args.filter)

        bb_to_print = self._format_output(bb_to_print, " Family   Backbone{fill}\n", 9)
        print(bb_to_print)
        return 0

    def _get_family(self, families: list) -> dict:
        assert isinstance(families, list)
        not_avail = []
        avail_print = ", ".join(list(self.available_backbone.keys()))

        retrieved = dict()
        for fam in families:
            available = True
            if fam not in self.available_backbone:
                not_avail.append(fam)
                available = False
            if available:
                retrieved[fam] = self.available_backbone[fam].copy()

        if not_avail:
            if len(not_avail) > 1:
                not_avail_print = ", ".join("'{}'".format(x) for x in not_avail[:-1])
                not_avail_print += ", and '{}'".format(not_avail[-1])
            else:
                not_avail_print = "'{}'".format(not_avail[0])
            print("{} family is not available.\nPlease choose from: {}\n"
                .format(not_avail_print, avail_print))
        return retrieved

    def _filter(self, data, pattern) -> dict:
        assert isinstance(data, dict)
        assert isinstance(pattern, str)

        retrieved = {n: fnmatch.filter(val, pattern) for n, val in data.items()}
        retrieved = {n: val for n,val in retrieved.items() if val}
        return retrieved


class ModelList(ListBase):
    available_model = {n.__name__: v for n,v in models.supported_models.copy().items()}

    def __init__(self):
        super().__init__()

    @classmethod
    def add_parser(cls, subparsers, parent_parser):
        HELP = "List available models"
        usage = "\n  vortex list model"
        parser = subparsers.add_parser(
            "model",
            description=HELP,
            help=HELP,
            formatter_class=argparse.RawTextHelpFormatter,
            usage=usage
        )

        parser.set_defaults(cmp_cls=cls)

    def run(self, args):
        formatted = self._format_output(self.available_model, "  Module     Name{fill}\n", 6, 12)
        print(formatted)

        return 0


class DatasetList(ListBase):

    def __init__(self):
        super().__init__()

    @classmethod
    def add_parser(cls, subparsers, parent_parser):
        HELP = "List available datasets"
        usage = "\n  vortex list dataset"
        parser = subparsers.add_parser(
            "dataset",
            description=HELP,
            parents=[parent_parser],
            help=HELP,
            formatter_class=argparse.RawTextHelpFormatter,
            usage=usage
        )

        parser.set_defaults(cmp_cls=cls)

    def run(self, args):
        from vortex.development.utils.data.dataset import dataset
        available = dataset.all_datasets.copy()
        formatted = self._format_output(available, "  Module     Name{fill}\n", 6, 12)
        print(formatted)

        return 0


class RuntimeList(ListBase):

    def __init__(self):
        super().__init__()

    @classmethod
    def add_parser(cls, subparsers, parent_parser):
        pass

    def run(self, args):
        pass


COMPONENTS = [
    BackboneList,
    ModelList,
    DatasetList
]


def main(args):
    component = args.cmp_cls()
    component.run(args)
    return 0


def add_parser(subparsers, parent_parser):
    LIST_HELP = "List available components in vortex"
    usage = "\n  vortex list <COMPONENT> [options]"
    parser = subparsers.add_parser(
        "list",
        parents=[parent_parser],
        description=LIST_HELP,
        help=LIST_HELP,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=usage
    )

    subparsers = parser.add_subparsers(
        title="Available Components",
        metavar="COMPONENT",
        dest="component",
        help="Use `vortex list <COMPONENT> --help` for command-specific help."
    )
    subparsers.required = True
    subparsers.dest = "component"

    for cmp in COMPONENTS:
        cmp.add_parser(subparsers, parent_parser)

    parser.set_defaults(func=main)

