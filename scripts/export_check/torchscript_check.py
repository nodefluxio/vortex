import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[2].joinpath("tests")))

from tests.test_torchscript_exporter import test_exporter, output_dir, model_argmap
from vortex.networks.modules.backbones import all_models as all_backbones


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-image', help='optional example image for tracing')
    parser.add_argument('--backbones', default=all_backbones, choices=all_backbones, 
        nargs='+', help='backbone(s) to test')
    parser.add_argument('--models', default=list(model_argmap.keys()), 
        choices=list(model_argmap.keys()), nargs='+', help='model(s) to test')
    parser.add_argument('--exclude-backbones', default=[], choices=all_backbones, 
        nargs='+', help='exclude this backbone(s) when testing')
    parser.add_argument('--exclude-models', default=[], choices=list(model_argmap.keys()), 
        nargs='+', help='model(s) to exclude')
    args = parser.parse_args()

    print("WARNING: this check might be storage and memory intensive")
    print("WARNING: all models are exported to {vortex_dir}/%s" % output_dir)
    if args.example_image is not None and not Path(args.example_image).exists():
        raise RuntimeError("example image %s not exist" % args.example_image)

    models = [m for m in args.models if m not in args.exclude_models]
    backbones = [bb for bb in args.backbones if bb not in args.exclude_backbones]
    n = 0
    total = len(models) * len(backbones)
    for model in models:
        for backbone in backbones:
            print("\n[{}/{}]\t".format(n, total), model, backbone)
            test_exporter(model, backbone=backbone, image=args.example_image, remove_output=False)
            n += 1
    print("\nDONE!")
