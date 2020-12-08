from pathlib import Path
proj_path = Path(__file__).parents[2]

from vortex.development.utils.parser.parser import load_config, check_config


def test_check_config():
    exp_path = Path(proj_path.joinpath("experiments", "configs"))
    cfg_path = [x for x in exp_path.iterdir() if not x.is_dir() and x.suffix == '.yml']
    for path in cfg_path:
        if path.name == 'resnet50_detr_coco.yml':
            continue
        config = load_config(str(path))
        check_result = check_config(config, experiment_type='train')
        assert check_result.valid, "%s is not valid, result:\n%s" % (path, str(check_result))
