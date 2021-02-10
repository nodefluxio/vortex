from _pytest.mark import param
import pytest
import torch
import pytorch_lightning as pl

from pathlib import Path
from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import DataLoader
from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator
from pytorch_lightning.callbacks import ProgressBarBase, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from vortex.development.pipelines.trainer import TrainingPipeline
from vortex.development import __version__ as vortex_version

from ..common import (
    DummyModel, DummyDataset, DummyDataModule,
    prepare_model, patched_pl_trainer,
    MINIMAL_TRAINER_CFG,
    state_dict_is_equal
)


@pytest.mark.parametrize(
    ('device', 'expected_gpu', 'expected_auto_select'),
    [
        pytest.param(None, None, False, id="on cpu"),
        pytest.param("cpu", None, False, id="on cpu with string"),
        pytest.param("cuda", 1, True, id="on gpu autoselect"),
        pytest.param("cuda:0", "0", False, id="on gpu 0"),
        pytest.param("cuda:0:1", "0", False, id="invalid device", marks=pytest.mark.xfail)
    ]
)
def test_args_device(tmp_path, device, expected_gpu, expected_auto_select):
    config = dict(device=device)
    expected = dict(gpus=expected_gpu, auto_select_gpus=expected_auto_select)

    kwargs = TrainingPipeline._trainer_args_device(config)
    assert kwargs == expected

    if torch.cuda.is_available() or expected_gpu is None:
        model = prepare_model(deepcopy(MINIMAL_TRAINER_CFG))
        trainer = patched_pl_trainer(str(tmp_path), model, trainer_args=kwargs)
        expected_accelerator = CPUAccelerator if expected_gpu is None else GPUAccelerator
        assert isinstance(trainer.accelerator_backend, expected_accelerator)
        assert trainer.data_parallel_device_ids == (None if expected_gpu is None else [0])
        assert trainer.num_gpus == (0 if expected_gpu is None else 1)


@pytest.mark.parametrize(
    "val_epoch", [1, 2]
)
def test_args_validation_interval(tmp_path, val_epoch):
    base_config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))

    cfg_val_interval = dict(validator=dict(val_epoch=val_epoch))
    expected = dict(check_val_every_n_epoch=val_epoch)
    kwargs = TrainingPipeline._trainer_args_validation_interval(cfg_val_interval)
    assert kwargs == expected

    config = deepcopy(base_config)
    config.update(cfg_val_interval)
    model = prepare_model(config)
    trainer = patched_pl_trainer(str(tmp_path), model, trainer_args=kwargs)
    assert trainer.check_val_every_n_epoch == val_epoch


    cfg_val_interval = dict(trainer=dict(validation_interval=val_epoch))
    kwargs = TrainingPipeline._trainer_args_validation_interval(cfg_val_interval)
    assert kwargs == expected

    config = deepcopy(base_config)
    config['trainer'].update(cfg_val_interval['trainer'])
    model = prepare_model(config)
    trainer = patched_pl_trainer(str(tmp_path), model, trainer_args=kwargs)
    assert trainer.check_val_every_n_epoch == val_epoch


def test_args_validation_interval_failed():
    with pytest.raises(ValueError):
        config = dict(validator=dict(val_epoch="1,2"))
        TrainingPipeline._trainer_args_validation_interval(config)

    with pytest.raises(ValueError):
        config = dict(trainer=dict(validation_interval="1,2"))
        TrainingPipeline._trainer_args_validation_interval(config)

    with pytest.raises(ValueError):
        config = dict(validator=dict(val_epoch=0))
        TrainingPipeline._trainer_args_validation_interval(config)

    with pytest.raises(ValueError):
        config = dict(trainer=dict(validation_interval=0))
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


@pytest.mark.parametrize(
    'accumulate',
    [
        1,
        2,
        pytest.param({0: 2}, id='dict simple'),
        pytest.param({2: 2, 10: 1}, id='dict more')
    ]
)
def test_args_accumulate_grad(tmp_path, accumulate):
    expected = dict(accumulate_grad_batches=accumulate)
    old_cfg = deepcopy(MINIMAL_TRAINER_CFG)
    new_cfg = deepcopy(MINIMAL_TRAINER_CFG)
    if accumulate:
        old_cfg['trainer'].update(driver=dict(args=dict(accumulation_step=accumulate)))
        new_cfg['trainer'].update(accumulate_step=accumulate)

    ## old cfg
    kwargs = TrainingPipeline._trainer_args_accumulate_grad(old_cfg)
    assert kwargs == expected
    model = prepare_model(old_cfg)
    trainer = patched_pl_trainer(str(tmp_path), model, trainer_args=kwargs)
    assert trainer.accumulate_grad_batches == accumulate

    ## new cfg
    kwargs = TrainingPipeline._trainer_args_accumulate_grad(new_cfg)
    assert kwargs == expected
    model = prepare_model(new_cfg)
    trainer = patched_pl_trainer(str(tmp_path), model, trainer_args=kwargs)
    assert trainer.accumulate_grad_batches == accumulate


@pytest.mark.parametrize('accumulate_type', ['int', 'dict'])
def test_args_accumulate_grad_fail(accumulate_type):
    old_cfg = deepcopy(MINIMAL_TRAINER_CFG)
    new_cfg = deepcopy(MINIMAL_TRAINER_CFG)

    accumulate = -2 if accumulate_type == 'int' else {0: -2}
    old_cfg['trainer'].update(driver=dict(args=dict(accumulation_step=accumulate)))
    new_cfg['trainer'].update(accumulate_step=accumulate)

    with pytest.raises(ValueError):
        TrainingPipeline._trainer_args_accumulate_grad(old_cfg)
    with pytest.raises(ValueError):
        TrainingPipeline._trainer_args_accumulate_grad(new_cfg)

    accumulate = None if accumulate_type == 'int' else {0: None}
    old_cfg['trainer'].update(driver=dict(args=dict(accumulation_step=accumulate)))
    new_cfg['trainer'].update(accumulate_step=accumulate)
    with pytest.raises(TypeError):
        TrainingPipeline._trainer_args_accumulate_grad(old_cfg)
    with pytest.raises(TypeError):
        TrainingPipeline._trainer_args_accumulate_grad(new_cfg)


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
        1, 2, 3,
        pytest.param(0, id="invalid value 0", marks=pytest.mark.xfail)
    ]
)
def test_checkpoint_save_epoch(tmp_path, save_epoch):
    config = deepcopy(MINIMAL_TRAINER_CFG)
    config['trainer'].update(dict(save_epoch=save_epoch))
    config = EasyDict(config)
    fname_fmt = "{}-epoch={{}}.pth".format(config['experiment_name'])
    ckpt_path = tmp_path.joinpath("version_0", "checkpoints")

    model = prepare_model(config)
    ckpt_callbacks = TrainingPipeline.create_model_checkpoints(config, model)
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=ckpt_callbacks)

    ## init all first
    for callback in ckpt_callbacks:
        callback.on_pretrain_routine_start(trainer, model)

    for callback in ckpt_callbacks:
        ## just check save epoch checkpoint callbacks
        if not hasattr(callback, 'save_epoch'):
            continue

        assert callback.period == save_epoch
        epoch_saved = []
        for _ in range(5):
            epoch = trainer.current_epoch
            callback.on_validation_end(trainer, model)

            if (epoch+1) % save_epoch == 0:
                epoch_saved.append(epoch)
                assert all(ckpt_path.joinpath(fname_fmt.format(e)).exists() for e in epoch_saved)
                fpath_this_epoch = ckpt_path.joinpath(fname_fmt.format(epoch))

                checkpoint = torch.load(fpath_this_epoch)
                assert checkpoint['config'] == dict(config)
                assert checkpoint['metrics'] == trainer.logger_connector.callback_metrics
                assert checkpoint['class_names'] == model.class_names
                assert checkpoint['vortex_version'] == vortex_version

                assert 'checkpoint_last' in checkpoint['callbacks']
                assert 'checkpoint_epoch' in checkpoint['callbacks']
                assert checkpoint['epoch'] == epoch+1
            else:
                assert all(ckpt_path.joinpath(fname_fmt.format(e)).exists() for e in epoch_saved)

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


@pytest.mark.parametrize(
    ("no_log", "logger_module"),
    [
        pytest.param(False, "tensorboard", id="tensorboard"),
        pytest.param(False, "TensorBoardLogger", id="TensorBoardLogger"),
        pytest.param(True, "tensorboard", id="no log"),
        pytest.param(False, "invalid", id="invalid logger", marks=pytest.mark.xfail)
    ]
)
def test_create_loggers(tmp_path, no_log, logger_module):
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))

    ## with bool logger
    config['trainer']['logger'] = True
    loggers = TrainingPipeline.create_loggers(str(tmp_path), config, no_log)
    assert loggers == (True if not no_log else False)

    config['trainer']['logger'] = False
    loggers = TrainingPipeline.create_loggers(str(tmp_path), config, no_log)
    assert loggers == False

    ## with logger no args
    config['trainer']['logger'] = dict(module=logger_module)
    loggers = TrainingPipeline.create_loggers(str(tmp_path), config, no_log)
    assert isinstance(loggers, pl.loggers.TensorBoardLogger) != no_log

    ## config in config.trainer.logger
    config['trainer']['logger'] = dict(
        module=logger_module,
        args=dict(name=config['experiment_name'])
    )
    loggers = TrainingPipeline.create_loggers(str(tmp_path), config, no_log)
    assert isinstance(loggers, pl.loggers.TensorBoardLogger) != no_log

    ## config in config.logging
    config['logging'] = config['trainer'].pop('logger')
    loggers = TrainingPipeline.create_loggers(str(tmp_path), config, no_log)
    assert isinstance(loggers, pl.loggers.TensorBoardLogger) != no_log

    model = prepare_model(config)
    trainer_args = dict(
        logger=loggers, checkpoint_callback=False,
        limit_train_batches=2, limit_val_batches=2,
        progress_bar_refresh_rate=0, max_epochs=1,
        num_sanity_val_steps=0, weights_summary=None,
    )
    trainer = patched_pl_trainer(str(tmp_path), model, gpus=None, trainer_args=trainer_args)
    datamodule = DummyDataModule()

    ## running sanity check, self.log is in model.training_step()
    trainer.fit(model, datamodule)

    if not no_log:
        experiment_dir = tmp_path.joinpath(loggers.name, 'version_0')
        assert any(f.match('events.out.tfevents.*') for f in experiment_dir.iterdir())

    ## config in config.logging with bool
    config['logging'] = True
    loggers = TrainingPipeline.create_loggers(str(tmp_path), config, no_log)
    assert loggers == (True if not no_log else False)


def test_create_loggers_failed(tmp_path):
    base_config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))

    ## 'module' cfg not found
    with pytest.raises(RuntimeError):
        config = deepcopy(base_config)
        config['logging'] = dict(
            invalid='tensorboard',
            args=dict(name=config['experiment_name'])
        )
        TrainingPipeline.create_loggers(str(tmp_path), config, no_log=False)

    ## logger name not found
    with pytest.raises(RuntimeError):
        config = deepcopy(base_config)
        config['logging'] = dict(
            module='nothing',
            args=dict(name=config['experiment_name'])
        )
        TrainingPipeline.create_loggers(str(tmp_path), config, no_log=False)

    ## invalid cfg type
    with pytest.raises(TypeError):
        config = deepcopy(base_config)
        config['logging'] = "invalid"
        TrainingPipeline.create_loggers(str(tmp_path), config, no_log=False)


def test_handle_resume_train(tmp_path):
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))

    ## no checkpoint in config and asked to resume
    with pytest.raises(RuntimeError):
        TrainingPipeline._handle_resume_checkpoint(config, resume=True)

    ## checkpoint None and asked to resume
    with pytest.raises(RuntimeError):
        config['checkpoint'] = None
        TrainingPipeline._handle_resume_checkpoint(config, resume=True)

    ## checkpoint not found
    with pytest.raises(RuntimeError):
        config['checkpoint'] = str(tmp_path.joinpath("ckpt.pth"))
        TrainingPipeline._handle_resume_checkpoint(config, resume=True)

    model = prepare_model(config)
    ckpt_callback = TrainingPipeline.create_model_checkpoints(config, model)[0]
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=[ckpt_callback])
    ckpt_callback.on_pretrain_routine_start(trainer, model)
    ckpt_callback.on_validation_end(trainer, model)
    fpath = tmp_path.joinpath("version_0", "checkpoints", config['experiment_name'] + "-last.pth")

    config['checkpoint'] = str(fpath)
    ckpt_path, state_dict = TrainingPipeline._handle_resume_checkpoint(config, resume=True)
    assert str(ckpt_path) == str(fpath)
    assert state_dict_is_equal(state_dict, model.cpu().state_dict())

    trainer_args = dict(resume_from_checkpoint=ckpt_path)
    trainer = patched_pl_trainer(str(tmp_path), model, trainer_args=trainer_args)
    trainer.checkpoint_connector.restore_weights(model)
    assert trainer.current_epoch == 1 and trainer.global_step == 1


def test_handle_resume_train_not_resume(tmp_path):
    ## when resume is False
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))

    ckpt_path, state_dict = TrainingPipeline._handle_resume_checkpoint(config, resume=False)
    assert ckpt_path is None and state_dict is None

    config['checkpoint'] = None
    ckpt_path, state_dict = TrainingPipeline._handle_resume_checkpoint(config, resume=False)
    assert ckpt_path is None and state_dict is None

    ## checkpoint path not found
    config['checkpoint'] = str(tmp_path.joinpath("ckpt.pth"))
    ckpt_path, state_dict = TrainingPipeline._handle_resume_checkpoint(config, resume=False)
    assert ckpt_path is None and state_dict is None

    ## checkpoint path is set and found
    model = prepare_model(config)
    ckpt_callback = TrainingPipeline.create_model_checkpoints(config, model)[0]
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=[ckpt_callback])
    ckpt_callback.on_pretrain_routine_start(trainer, model)
    ckpt_callback.on_validation_end(trainer, model)
    fpath = tmp_path.joinpath("version_0", "checkpoints", config['experiment_name'] + "-last.pth")

    config['checkpoint'] = str(fpath)
    ckpt_path, state_dict = TrainingPipeline._handle_resume_checkpoint(config, resume=False)
    assert ckpt_path == None
    assert state_dict_is_equal(state_dict, model.cpu().state_dict())


def test_handle_resume_verify_config(tmp_path):
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))
    model = prepare_model(config)
    ckpt_callback = TrainingPipeline.create_model_checkpoints(config, model)[0]
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=[ckpt_callback])
    ckpt_callback.on_pretrain_routine_start(trainer, model)
    ckpt_callback.on_validation_end(trainer, model)
    fpath = tmp_path.joinpath("version_0", "checkpoints", config['experiment_name'] + "-last.pth")

    ## normal run
    config['checkpoint'] = str(fpath)
    ckpt_path, state_dict = TrainingPipeline._handle_resume_checkpoint(config, resume=True)
    assert str(ckpt_path) == str(fpath)
    assert state_dict_is_equal(state_dict, model.cpu().state_dict())

    ## different model name
    with pytest.raises(RuntimeError):
        cfg_different_model = deepcopy(config)
        cfg_different_model['model']['name'] = 'DifferentModel'
        TrainingPipeline._handle_resume_checkpoint(cfg_different_model, resume=True)

    ## different network args
    with pytest.raises(RuntimeError):
        cfg_different_model_args = deepcopy(config)
        cfg_different_model_args['model']['network_args'] = dict(num_classes=2)
        TrainingPipeline._handle_resume_checkpoint(cfg_different_model_args, resume=True)

    ## different dataset name
    with pytest.raises(RuntimeError):
        cfg_different_dataset = deepcopy(config)
        cfg_different_dataset['dataset']['train']['name'] = 'DifferentDataset'
        TrainingPipeline._handle_resume_checkpoint(cfg_different_dataset, resume=True)

    ## dataset config not found
    with pytest.raises(RuntimeError):
        cfg_not_found_dataset = deepcopy(config)
        cfg_not_found_dataset['dataset'].pop('train')
        TrainingPipeline._handle_resume_checkpoint(cfg_not_found_dataset, resume=True)


@pytest.mark.parametrize(
    "output_dir",
    [
        pytest.param(False, id="no output dir"),
        pytest.param(None, id="output dir None"),
        "experiments",
        "~/vortex/experiments"
    ]
)
def test_format_experiment_dir(output_dir):
    config = deepcopy(MINIMAL_TRAINER_CFG)
    if output_dir is not False:
        config['output_directory'] = output_dir
    if not output_dir:
        output_dir = "experiments/outputs"

    exp_dir = TrainingPipeline._format_experiment_dir(config)
    assert isinstance(exp_dir, Path)
    assert str(exp_dir) == "{}/{}".format(output_dir, config['experiment_name'])


@pytest.mark.parametrize(
    ("device", "hypopt"),
    [
        pytest.param('cpu', False, id="normal model on cpu"),
        pytest.param(
            'cuda:0', False, id="normal mode on gpu",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
        ),
        pytest.param('cpu', True, id="hypopt mode")
    ]
)
def test_create_trainer(tmp_path, caplog, device, hypopt):
    caplog.set_level(20)    ## set log level to INFO
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))
    config['device'] = device
    config['trainer']['validation_interval'] = 2
    config['trainer']['seed'] = 1395
    config['trainer']['logger'] = True

    model = prepare_model(config)
    trainer = TrainingPipeline.create_trainer(str(tmp_path), config, model, hypopt=hypopt)

    assert sum(1 for rec in caplog.records if rec.module == 'trainer') == 1     ## log from seed everything
    assert trainer.max_epochs == config['trainer']['epoch']
    assert trainer._default_root_dir == str(tmp_path)
    assert trainer._weights_save_path == str(tmp_path)
    assert trainer.weights_summary == None
    assert trainer.benchmark == False
    assert trainer.deterministic == True
    assert trainer.check_val_every_n_epoch == 2

    ## model checkpoint, lr monitor and progress bar callback if not hypopt else pbar only
    assert len(trainer.callbacks) == (1 if hypopt else 3)
    ## model checkpoint is not available when hypopt
    assert any(isinstance(c, ModelCheckpoint) for c in trainer.callbacks) != hypopt
    ## progress bar callback is always available
    assert any(isinstance(c, ProgressBarBase) for c in trainer.callbacks)
    ## loggers is not available when hypopt
    assert isinstance(trainer.logger, TensorBoardLogger) != hypopt
    assert any(isinstance(c, LearningRateMonitor) for c in trainer.callbacks) != hypopt

    ## training device
    trainer.accelerator_backend = trainer.accelerator_connector.select_accelerator()
    trainer.accelerator_backend.setup(model)
    expected_accelerator = CPUAccelerator if device == 'cpu' else GPUAccelerator
    assert isinstance(trainer.accelerator_backend, expected_accelerator)
    assert trainer.data_parallel_device_ids == (None if device == 'cpu' else [0])
    assert trainer.num_gpus == (0 if device == 'cpu' else 1)


def test_create_trainer_with_args(tmp_path):
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))
    config['trainer']['seed'] = 1395

    model = prepare_model(config)
    trainer = TrainingPipeline.create_trainer(str(tmp_path), config, model, hypopt=False)
    assert trainer.max_epochs == config['trainer']['epoch']
    assert trainer._default_root_dir == str(tmp_path)
    assert trainer._weights_save_path == str(tmp_path)
    assert trainer.weights_summary == None
    assert trainer.benchmark == False
    assert trainer.deterministic == True

    config['trainer']['args'] = dict(benchmark=True, log_gpu_memory="min_max", auto_lr_find=True)
    model = prepare_model(config)
    trainer = TrainingPipeline.create_trainer(str(tmp_path), config, model, hypopt=False)
    assert trainer.benchmark == True
    assert trainer.log_gpu_memory == "min_max"
    assert trainer.auto_lr_find == True

    with pytest.raises(TypeError):
        config['trainer']['args'] = "invalid"
        TrainingPipeline.create_trainer(str(tmp_path), config, model, hypopt=False)


def test_training_pipeline_init(tmp_path):
    base_config_path = "tests/config/test_classification_pipelines.yml"

    ## edit config
    config = TrainingPipeline._get_config(base_config_path)
    config['model']['network_args']['backbone'] = 'resnet18'
    config_path = TrainingPipeline._dump_config(config)

## TODO: other vortex behavior (?)
