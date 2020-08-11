from .version import __version__
from .runtime_map import model_runtime_map
from .factory import create_runtime_model

def check_available_runtime():
    for name, runtime_map in model_runtime_map.items() :
        if name =='pt':
            name = 'torchscript'
        for rt in runtime_map:
            print('Runtime {} <{}>: {}'.format(
                name, runtime_map[rt].__name__, 'available' \
                    if runtime_map[rt].is_available() else 'unavailable'
            ))