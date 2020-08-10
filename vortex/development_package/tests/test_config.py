import os
import sys
from pathlib import Path
proj_path = os.path.abspath(Path(__file__).parents[1])
sys.path.append(proj_path)

from vortex.development.utils.parser.parser import load_config, check_config


def test_check_config():
    exp_path = Path(os.path.join(proj_path, "experiments", "configs"))
    cfg_path = [x for x in exp_path.iterdir() if not x.is_dir() and x.suffix == '.yml']
    for path in cfg_path:
        path = os.path.abspath(path)
        config = load_config(path)
        check_result = check_config(config, experiment_type='train')
        assert check_result.valid, "%s is not valid, result:\n%s" % (path, str(check_result))
    

if __name__ == "__main__":
    test_check_config()
