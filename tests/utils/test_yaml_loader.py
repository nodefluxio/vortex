import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2].joinpath('src', 'development')))

import yaml

from vortex.development.utils.parser.loader import Loader
from vortex.development.utils.parser.parser import load_config, check_config

def test_loader() :
    test_file = 'tests/config/test1.yml'
    with open(test_file) as f :
        data = yaml.load(f, Loader=Loader)
    assert isinstance(data, dict)
    assert data['name']['cfg']=='config'

def test_check_config_train() :
    test_file = 'tests/config/broken/model.yml'
    config = load_config(test_file)
    check_result = check_config(config, experiment_type='train')
    assert not check_result.valid