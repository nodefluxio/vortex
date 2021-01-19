import pytest
import torch

from pathlib import Path
from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import DataLoader

from vortex.development.pipelines.trainer import TrainingPipeline
from vortex.development import __version__ as vortex_version

from ..common import (
    DummyModel, DummyDataset,
    prepare_model, patched_pl_trainer,
    MINIMAL_TRAINER_CFG
)


@pytest.mark.parametrize(
    ('device', 'expected_gpu', 'expected_auto_select'),
    [
        pytest.param(None, None, False, id="on cpu"),
        pytest.param("cuda", 1, True, id="on gpu autoselect"),
        pytest.param("cuda:1", "1", False, id="on gpu 1")
    ]
)
def test_args_device(device, expected_gpu, expected_auto_select):
    config = dict(device=device)
    expected = dict(gpus=expected_gpu, auto_select_gpus=expected_auto_select)

    kwargs = TrainingPipeline._trainer_args_device(config)
    assert kwargs == expected


def test_args_validation_interval():
    val_epoch = 1
    config = dict(validator=dict(val_epoch=val_epoch))
    expected = dict(check_val_every_n_epoch=val_epoch)
    kwargs = TrainingPipeline._trainer_args_validation_interval(config)
    assert kwargs == expected

    config = dict(trainer=dict(validate_interval=val_epoch))
    kwargs = TrainingPipeline._trainer_args_validation_interval(config)
    assert kwargs == expected

    with pytest.raises(RuntimeError):
        config = dict(validator=dict(val_epoch="1,2"))
        TrainingPipeline._trainer_args_validation_interval(config)

    with pytest.raises(RuntimeError):
        config = dict(trainer=dict(validate_interval="1,2"))
        TrainingPipeline._trainer_args_validation_interval(config)


def test_args_set_seed(caplog, recwarn):
    caplog.set_level(20)    ## set log level to INFO
    deterministic, benchmark = True, True
    expected_ret = dict(deterministic=deterministic, benchmark=benchmark)
    seed = 1395
    config = dict(
        seed=dict(
            cudnn=dict(deterministic=deterministic, benchmark=benchmark),
            torch=seed,
            numpy=seed
        )
    )

    ## deprecated seed config
    with pytest.warns(DeprecationWarning):
        kwargs = TrainingPipeline._trainer_args_set_seed(deepcopy(config))
        assert kwargs == expected_ret

    ## deprecated 'seed.cudnn'
    with pytest.warns(DeprecationWarning):
        config['trainer'] = dict(seed=config.pop('seed'))
        kwargs = TrainingPipeline._trainer_args_set_seed(deepcopy(config))
        assert kwargs == expected_ret

    ## normal run
    recwarn.clear()
    config['trainer']['seed'].update(config['trainer']['seed'].pop('cudnn'))
    kwargs = TrainingPipeline._trainer_args_set_seed(deepcopy(config))
    assert len(recwarn) == 0
    assert kwargs == expected_ret

    ## seed everything
    recwarn.clear()
    caplog.clear()
    expected_ret = dict(deterministic=True, benchmark=False)
    config = dict(trainer=dict(seed=seed))
    kwargs = TrainingPipeline._trainer_args_set_seed(deepcopy(config))
    assert kwargs == expected_ret
    assert len(recwarn) == 0
    assert len(caplog.records) == 1

    ## unknown type 
    with pytest.raises(RuntimeError):
        config = dict(trainer=dict(seed="this is invalid"))
        TrainingPipeline._trainer_args_set_seed(deepcopy(config))


def test_get_config():
    config_path = "tests/config/test_classification_pipelines.yml"

    ## test on str
    config = TrainingPipeline._get_config(config_path)
    assert isinstance(config, EasyDict)

    ## test on Path
    config = TrainingPipeline._get_config(Path(config_path))
    assert isinstance(config, EasyDict)

    ## test on loaded config
    config = TrainingPipeline._get_config(config)
    assert isinstance(config, EasyDict)


def test_check_config(caplog):
    config_path = "tests/config/test_classification_pipelines.yml"
    config = TrainingPipeline._get_config(config_path)

    ## normal
    TrainingPipeline._check_experiment_config(config)

    ## invalid validation -> raising warning
    cfg_val = deepcopy(config)
    cfg_val['dataset'].pop('eval')
    TrainingPipeline._check_experiment_config(cfg_val)
    assert len(caplog.records) > 0 
    assert caplog.records[0].levelname == "WARNING"

    ## invalid train -> raise error
    with pytest.raises(RuntimeError):
        cfg_val = deepcopy(config)
        cfg_val['dataset'].pop('train')
        TrainingPipeline._check_experiment_config(cfg_val)


def test_dump_config(tmp_path):
    config_path = "tests/config/test_classification_pipelines.yml"
    config = TrainingPipeline._get_config(config_path)

    dumped_cfg_path = TrainingPipeline._dump_config(config, tmp_path)
    assert dumped_cfg_path.exists()
    assert str(dumped_cfg_path) == str(tmp_path.joinpath("config.yml"))
    dumped_config = TrainingPipeline._get_config(dumped_cfg_path)
    assert dumped_config == config


def test_copy_data_to_model():
    config = deepcopy(MINIMAL_TRAINER_CFG)
    model = DummyModel(num_classes=5)
    dataloader = DataLoader(DummyDataset(), batch_size=4)

    assert model.config is None and model.class_names is None

    TrainingPipeline._copy_data_to_model(dataloader, config, model)

    assert model.config == config
    assert model.class_names == dataloader.dataset.class_names


def test_checkpoint_default_last(tmp_path):
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))

    model = prepare_model(config)
    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(config, model)
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=ckpt_callbacks)

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
    config = deepcopy(MINIMAL_TRAINER_CFG)
    config['trainer'].update(dict(save_best_metrics=deepcopy(save_best)))
    config = EasyDict(config)

    model = prepare_model(config)
    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(config, model)
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=ckpt_callbacks)

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
        pytest.param(0, id="invalid value 0", marks=pytest.mark.xfail)
    ]
)
def test_checkpoint_save_epoch(tmp_path, save_epoch):
    config = deepcopy(MINIMAL_TRAINER_CFG)
    config['trainer'].update(dict(save_epoch=save_epoch))
    config = EasyDict(config)

    model = prepare_model(config)
    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(config, model)
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=ckpt_callbacks)

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


@pytest.mark.parametrize(
    ('train', 'eval'),
    [
        pytest.param(True, True, id='train and eval'),
        pytest.param(True, False, id='no eval'),
        pytest.param(
            False, True, id='no train',
            marks=pytest.mark.xfail(reason='dataset train is required')
        )
    ]
)
def test_create_dataloaders(train, eval):
    config = EasyDict(dict(
        model=dict(preprocess_args=dict(
            input_size=224,
            input_normalization=dict(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        )),
        dataset=dict(
            train=dict(
                name='ImageFolder',
                args=dict(root='tests/test_dataset/classification/train')
            ),
            eval=dict(
                name='ImageFolder',
                args=dict(root='tests/test_dataset/classification/val')
            )
        ),
        dataloader=dict(
            module="PytorchDataLoader",
            args=dict(num_workers=1, batch_size=2, shuffle=True),
        ),
    ))
    if not train:
        config['dataset'].pop('train')
    if not eval:
        config['dataset'].pop('eval')

    model = prepare_model(config)
    train_loader, val_loader = TrainingPipeline.create_dataloaders(config, model)

    assert isinstance(train_loader, DataLoader)
    assert train_loader.num_workers == 1 and train_loader.batch_size == 2
    if eval:
        assert isinstance(val_loader, DataLoader)
        assert val_loader.num_workers == 1 and val_loader.batch_size == 2
    else:
        assert val_loader is None


def test_resume_train():
    assert 0

def test_create_loggers():
    assert 0


## TODO:
# - test on resume
# - test dataloader
# - test logger
# - other vortex behavior
