import os
import glob
import argparse
import torch

from pathlib import Path
from easydict import EasyDict as edict

from vortex.utils.parser import load_config, check_config
from vortex.core import engine
from vortex.core.factory import create_model, create_dataset

def update_checkpoints(config, model_paths):
    assert isinstance(model_paths, list) and isinstance(model_paths[0], str)
    assert isinstance(config, (str, edict))

    model_components = create_model(model_config=config.model)
    trainer = engine.create_trainer(
        config.trainer, experiment_logger=None,
        criterion=model_components.loss,
        model=model_components.network
    )
    dataset = create_dataset(config.dataset, config.model.preprocess_args, 'train')

    checkpoint = {
        "config": config,
        "class_names": class_names,
        # since we dont know how it trained, just put an empty optimizer state
        "optimizer_state": trainer.optimizer.state_dict(), 
    }
    for model_path in model_paths:
        print("=> processing {}".format(model_path))
        if not os.path.exists(model_path) and os.path.splitext(model_path)[-1] == '.pth':
            raise RuntimeError("Model path {} is invalid, make sure file is available "
                "and filename have extension of '.pth'".format(model_path))
        ckpt = torch.load(model_path)
        if all((k in ckpt) for k in ('epoch', 'state_dict', 'class_names', 'config')):
            continue 

        fname = os.path.split(model_path)[-1]
        epoch = config.trainer.epoch
        if 'epoch' in fname:
            epoch = int(fname.replace('.pth', '').split('-')[-1])
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        checkpoint.update({
            'epoch': epoch,
            'stated_dict': state_dict
        })
        torch.save(checkpoint, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str,
        help="configuration file path (.yml)")
    parser.add_argument('-w', '--weights', required=True, nargs='+',
        help="model's weights path, support multiple files")
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

    update_checkpoints(config, model_paths)
