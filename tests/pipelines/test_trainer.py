import pytest
import torch
import pytorch_lightning as pl

from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import DataLoader

from vortex.development.pipelines.trainer import TrainingPipeline
from vortex.development import __version__ as vortex_version

from ..common import DummyModel, DummyDataset


_REQUIRED_TRAINER_CFG = {
    'experiment_name': 'dummy_experiment',
    'device': 'cuda:0',
    'trainer': {
        'optimizer': {
            'method': 'SGD',
            'args': {'lr': 0.001}
        },
        'epoch': 2
    }
}


def patched_pl_trainer(experiment_dir, model, callbacks):
    TrainingPipeline._patch_trainer_components()
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        default_root_dir=experiment_dir,
        callbacks=callbacks
    )
    TrainingPipeline._patch_trainer_object(trainer)

    ## setup accelerator
    trainer.accelerator_backend = trainer.accelerator_connector.select_accelerator()
    trainer.accelerator_backend.setup(model)
    trainer.accelerator_backend.train_loop = trainer.train
    trainer.accelerator_backend.validation_loop = trainer.run_evaluation
    trainer.accelerator_backend.test_loop = trainer.run_evaluation

    ## dummy metrics data
    metrics = {
        'train_loss': torch.tensor(1.0891),
        'accuracy': torch.tensor(0.7618)
    }
    trainer.logger_connector.callback_metrics = metrics
    return trainer

def prepare_model(config, num_classes=5):
    model = DummyModel(num_classes=num_classes)
    model.config = config
    model.class_names = ["label_"+str(n) for n in range(num_classes)]
    return model


@pytest.mark.parametrize(
    ('device', 'expected_gpu', 'expected_auto_select'),
    [
        pytest.param(None, None, False, id="on cpu"),
        pytest.param("cuda", 1, True, id="on gpu autoselect"),
        pytest.param("cuda:1", "1", False, id="on gpu 1")
    ]
)
def test_decide_device(device, expected_gpu, expected_auto_select):
    config = dict(device=device)
    expected = dict(gpus=expected_gpu, auto_select_gpus=expected_auto_select)

    kwargs = TrainingPipeline._decide_device_to_use(config)
    assert kwargs == expected


def test_handle_validation_interval():
    val_epoch = 1

    config = dict(validator=dict(val_epoch=val_epoch))
    expected = dict(check_val_every_n_epoch=val_epoch)
    kwargs = TrainingPipeline._handle_validation_interval(config)
    assert kwargs == expected

    config = dict(trainer=dict(validate_interval=val_epoch))
    kwargs = TrainingPipeline._handle_validation_interval(config)
    assert kwargs == expected

    with pytest.raises(RuntimeError):
        config = dict(validator=dict(val_epoch="1,2"))
        TrainingPipeline._handle_validation_interval(config)

    with pytest.raises(RuntimeError):
        config = dict(trainer=dict(validate_interval="1,2"))
        TrainingPipeline._handle_validation_interval(config)


def test_copy_data_to_model():
    config = deepcopy(_REQUIRED_TRAINER_CFG)
    model = DummyModel(num_classes=5)
    dataloader = DataLoader(DummyDataset(), batch_size=4)

    assert model.config is None and model.class_names is None

    TrainingPipeline._copy_data_to_model(dataloader, config, model)

    assert model.config == config
    assert model.class_names == dataloader.dataset.class_names


def test_checkpoint_default_last(tmp_path):
    config = EasyDict(deepcopy(_REQUIRED_TRAINER_CFG))

    model = prepare_model(config)
    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(config, model)
    trainer = patched_pl_trainer(str(tmp_path), model, ckpt_callbacks)

    ckpt_callbacks[0].on_pretrain_routine_start(trainer, model)
    ckpt_callbacks[0].on_validation_end(trainer, model)

    fname = config['experiment_name'] + "-last.pth"
    fpath = tmp_path.joinpath("version_0", "checkpoints", fname)
    assert fpath.exists()

    ## check saved checkpoint
    checkpoint = torch.load(fpath)
    assert checkpoint['config'] == dict(config)
    assert checkpoint['metrics'] == trainer.logger_connector.callback_metrics
    assert checkpoint['class_names'] == model.class_names
    assert checkpoint['vortex_version'] == vortex_version
    assert 'checkpoint_last' in checkpoint['callbacks']


@pytest.mark.parametrize(
    "save_best",
    [
        'accuracy',
        'train_loss',
        pytest.param(['accuracy'], id='accuracy on list'),
        pytest.param(['accuracy', 'train_loss'], id='multiple checkpoint')
    ]
)
def test_checkpoint_save_best(tmp_path, save_best):
    config = deepcopy(_REQUIRED_TRAINER_CFG)
    config['trainer'].update(dict(save_best_metrics=deepcopy(save_best)))
    config = EasyDict(config)

    model = prepare_model(config)
    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(config, model)
    trainer = patched_pl_trainer(str(tmp_path), model, ckpt_callbacks)

    if isinstance(save_best, str):
        save_best = [save_best]
    ## default save last
    save_best.insert(0, None)
    assert ckpt_callbacks[0].monitor == None
    assert len(ckpt_callbacks) == len(save_best)+1

    ## init all first
    for callback in ckpt_callbacks:
        callback.on_pretrain_routine_start(trainer, model)

    for callback, monitor in zip(ckpt_callbacks, save_best):
        assert callback.monitor == monitor
        if monitor is None:
            continue

        delta = 0.2
        opt_strategy = model.available_metrics[monitor]
        original_metrics_val = deepcopy(trainer.logger_connector.callback_metrics)
        prev_metrics_val = deepcopy(trainer.logger_connector.callback_metrics)

        for action in (None, torch.add, torch.sub):
            ## update metrics
            if action is not None:
                metrics = original_metrics_val
                trainer.logger_connector.callback_metrics = {n: action(v, delta) for n, v in metrics.items()}

            callback.on_validation_end(trainer, model)

            fname = "{}-best_{}.pth".format(config['experiment_name'], monitor)
            fpath = tmp_path.joinpath("version_0", "checkpoints", fname)
            assert fpath.exists()

            if (opt_strategy == 'max' and action == torch.add) or (opt_strategy == 'min' and action == torch.sub):
                saved_metrics = trainer.logger_connector.callback_metrics
            else:
                saved_metrics = prev_metrics_val 

            checkpoint = torch.load(fpath)
            assert checkpoint['config'] == dict(config)
            assert checkpoint['metrics'] == saved_metrics
            assert checkpoint['class_names'] == model.class_names
            assert checkpoint['vortex_version'] == vortex_version

            ckpt_monitor_key = 'checkpoint_' + monitor
            assert ckpt_monitor_key in checkpoint['callbacks']
            assert checkpoint['callbacks'][ckpt_monitor_key]['monitor'] == monitor
            assert checkpoint['callbacks'][ckpt_monitor_key]['best_model_score'] == saved_metrics[monitor]

            ## step
            trainer.current_epoch += 1
            trainer.global_step += 2
            prev_metrics_val = deepcopy(trainer.logger_connector.callback_metrics)


@pytest.mark.parametrize(
    "save_epoch",
    [
        1, 2,
        pytest.param(0, marks=pytest.mark.xfail(reason="invalid value 0"))
    ]
)
def test_checkpoint_save_epoch(tmp_path, save_epoch):
    config = deepcopy(_REQUIRED_TRAINER_CFG)
    config['trainer'].update(dict(save_epoch=save_epoch))
    config = EasyDict(config)

    model = prepare_model(config)
    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(config, model)
    trainer = patched_pl_trainer(str(tmp_path), model, ckpt_callbacks)

    ## init all first
    for callback in ckpt_callbacks:
        callback.on_pretrain_routine_start(trainer, model)

    for callback in ckpt_callbacks:
        if not hasattr(callback, 'save_epoch'):
            continue

        assert callback.period == save_epoch

        for _ in range(2):
            epoch = trainer.current_epoch
            callback.on_validation_end(trainer, model)

            if (epoch+1) % save_epoch == 0:
                fname = "{}-epoch={}.pth".format(config['experiment_name'], epoch)
                fpath = tmp_path.joinpath("version_0", "checkpoints", fname)
                assert fpath.exists()

                checkpoint = torch.load(fpath)
                assert checkpoint['config'] == dict(config)
                assert checkpoint['metrics'] == trainer.logger_connector.callback_metrics
                assert checkpoint['class_names'] == model.class_names
                assert checkpoint['vortex_version'] == vortex_version

                assert 'checkpoint_last' in checkpoint['callbacks']
                assert 'checkpoint_epoch' in checkpoint['callbacks']
                assert checkpoint['epoch'] == epoch+1

            ## step
            trainer.current_epoch += 1
            trainer.global_step += 2


## TODO: test metrics

## TODO: test logger

## TODO: test created model

## TODO: other vortex behavior
