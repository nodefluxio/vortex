import os
import glob
import argparse
import torch

from pathlib import Path
from easydict import EasyDict as edict

if __name__ == "__main__":
    proj_path = Path(__file__).parents[1]
    sys.path.append(proj_path.joinpath('src/development'))

from vortex.development.utils.parser import load_config, check_config
from vortex.development.core import engine
from vortex.development.core.factory import create_model, create_dataset

def update_checkpoints(config, model_paths, override=False):
    assert isinstance(model_paths, list) and isinstance(model_paths[0], str)
    assert isinstance(config, (str, edict))

    model_components = create_model(model_config=config.model,stage='validate')
    trainer = engine.create_trainer(
        config.trainer, experiment_logger=None,
        criterion=model_components.loss,
        model=model_components.network
    )
    dataset = create_dataset(config.dataset, config.model.preprocess_args, 'train')

    checkpoint = {
        "config": config,
        "class_names": dataset.class_names,
        # since we dont know how it trained, just put an empty optimizer state
        "optimizer_state": trainer.optimizer.state_dict(), 
    }
    for idx, model_path in enumerate(model_paths):
        fdir, fname = os.path.split(model_path)
        updated_fname = fname
        if not override:
            basename, ext = os.path.splitext(updated_fname)
            updated_fname = basename + '_updated' + ext
        print("[{}/{}] updating {} to {}".format(idx+1, len(model_paths), 
            model_path, os.path.join(fdir, updated_fname)))

        if not os.path.exists(model_path) and os.path.splitext(model_path)[-1] == '.pth':
            raise RuntimeError("Model path {} is invalid, make sure file is available "
                "and filename have extension of '.pth'".format(model_path))
        ckpt = torch.load(model_path)
        if all((k in ckpt) for k in ('epoch', 'state_dict', 'class_names', 'config')):
            print("=> skipping {}, checkpoint already in new format")
            continue 

        epoch = config.trainer.epoch
        if 'epoch' in fname:
            epoch = fname.replace('.pth', '').split('-')[-1]
            epoch = int(''.join(d for d in epoch if d.isdigit()))
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        checkpoint.update({
            'epoch': epoch,
            'state_dict': state_dict
        })

        torch.save(checkpoint, os.path.join(fdir, updated_fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="update old model checkpoint to the new format")
    parser.add_argument('-c', '--config', required=True, type=str,
        help="configuration file path (.yml)")
    parser.add_argument('-w', '--weights', required=True, nargs='+',
        help="model's weights path, support multiple files and also wildcard with (*) char")
    parser.add_argument('--override', action='store_true', 
        help='override the original model by the updated instead of copy.')
    args = parser.parse_args()

    config = load_config(args.config)
    check_result = check_config(config, 'train')
    if not check_result.valid:
        raise RuntimeError("Invalid config {}".format(check_result))

    model_paths = []
    for model_path in args.weights:
        if '*' not in model_path:
            model_paths.append(model_path)
        else:
            model_paths.extend(glob.glob(model_path))

    model_paths = list(set(model_paths)) ## avoid multiple definition of the same file
    update_checkpoints(config, model_paths, override=args.override)
