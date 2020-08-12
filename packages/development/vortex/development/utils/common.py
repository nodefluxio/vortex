from pathlib import Path
from easydict import EasyDict
import shutil
from typing import Union

def check_and_create_output_dir(config : EasyDict,
                                experiment_logger = None,
                                config_path : Union[str,Path,None] = None):

    # Set base output directory
    base_output_directory = Path('experiments/outputs')
    if 'output_directory' in config:
        base_output_directory = Path(config.output_directory)

    # Set experiment directory
    experiment_directory = Path(base_output_directory/config.experiment_name)
    if not experiment_directory.exists():
        experiment_directory.mkdir(exist_ok=True, parents=True)

    # Set run directory
    run_directory = None
    if experiment_logger:
        run_directory=Path(experiment_directory/experiment_logger.run_key)
        if not run_directory.exists():
            run_directory.mkdir(exist_ok=True, parents=True)
        # Duplicate experiment config if specified to run directory
        if config_path:
            shutil.copy(config_path,str(run_directory/'config.yml'))

    return experiment_directory,run_directory