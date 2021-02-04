import pytest

from copy import deepcopy

from vortex.development.pipelines.trainer.progress import VortexProgressBar

from ..common import prepare_model, patched_pl_trainer, MINIMAL_TRAINER_CFG


@pytest.mark.parametrize("refresh_rate", [0, 1, 2, 3])
def test_progress_bar_refresh(tmp_path, refresh_rate):

    class CustomTestProgressBar(VortexProgressBar):

        train_batches_seen = 0
        val_batches_seen = 0
        test_batches_seen = 0

        def _check_metrics(self):
            assert self.metrics_bar is not None
            metrics_dict = self.trainer.progress_bar_metrics
            metrics_str = "  ".join("{}: {:.4g}".format(n, v) for n,v in metrics_dict.items())
            assert self.metrics_bar._fields['metrics'] == metrics_str

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
                assert self.train_progress_bar.count == self.train_batch_idx
                self._check_metrics()
            elif self.is_enabled and self.train_batch_idx < self.train_progress_bar.total:
                ## when not updated should be different count
                assert self.train_progress_bar.count != self.train_batch_idx
            if self.is_enabled and self.train_batch_idx >= self.train_progress_bar.total:
                ## closed and cleared on last batch
                assert all(self.train_progress_bar != c for c in self.manager.counters)
            self.train_batches_seen += 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if self.is_enabled and self.val_batch_idx % self.refresh_rate == 0:
                assert self.val_progress_bar.count == self.val_batch_idx
                if not self.trainer.running_sanity_check:
                    self._check_metrics()
            elif self.is_enabled and self.val_batch_idx < self.val_progress_bar.total:
                assert self.val_progress_bar.count != self.val_batch_idx
            self.val_batches_seen += 1

        def on_validation_end(self, trainer, pl_module):
            super().on_validation_end(trainer, pl_module)
            if not trainer.running_sanity_check:
                self._check_metrics()
                ## closed and cleared on validation end
                assert all(self.val_progress_bar != c for c in self.manager.counters)

        def on_fit_end(self, trainer, pl_module):
            super().on_fit_end(trainer, pl_module)
            ## manager is stop on finished training
            assert not self.manager.enabled

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if self.is_enabled and self.test_batch_idx % self.refresh_rate == 0:
                assert self.test_progress_bar.count == self.test_batch_idx
                self._check_metrics()
            elif self.is_enabled and self.test_batch_idx < self.test_progress_bar.total:
                assert self.test_progress_bar.count != self.test_batch_idx
            self.test_batches_seen += 1

    config = deepcopy(MINIMAL_TRAINER_CFG)
    model = prepare_model(config, model_args=dict(num_data=10))

    progress_bar = CustomTestProgressBar("TRAIN", refresh_rate=refresh_rate)
    trainer_args = dict(
        max_epochs=2,
        progress_bar_refresh_rate=101   ## should not affect the provided pbar
    )
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=[progress_bar], trainer_args=trainer_args)
    assert trainer.progress_bar_callback == progress_bar
    assert trainer.progress_bar_callback.refresh_rate == refresh_rate

    trainer.fit(model)
    assert progress_bar.train_batches_seen == 2 * progress_bar.total_train_batches
    assert progress_bar.val_batches_seen == 2 * progress_bar.total_val_batches + trainer.num_sanity_val_steps

    trainer.test(model)
    assert progress_bar.test_batches_seen == progress_bar.total_test_batches


@pytest.mark.parametrize(
    ("val_epoch", "disable_validation"),
    [
        pytest.param(1, False, id="val epoch 1"),
        pytest.param(2, False, id="val epoch 2"),
        pytest.param(3, False, id="val epoch 3"),
        pytest.param(1, True, id="val disabled"),
    ]
)
def test_progress_bar_update_epoch(tmp_path, val_epoch, disable_validation):
    val_epoch = 1
    disable_validation = False

    class CustomTestProgressBar(VortexProgressBar):
        def on_epoch_end(self, trainer, pl_module):
            super().on_epoch_end(trainer, pl_module)
            assert self.main_progress_bar.count == trainer.current_epoch + 1

    config = deepcopy(MINIMAL_TRAINER_CFG)
    model = prepare_model(config)

    progress_bar = CustomTestProgressBar("TRAIN")
    trainer_args = dict(
        max_epochs=5,
        progress_bar_refresh_rate=101,   ## should not affect the provided pbar
        check_val_every_n_epoch=val_epoch,
    )
    if disable_validation:
        trainer_args['limit_val_batches'] = 0
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=[progress_bar], trainer_args=trainer_args)
    assert trainer.progress_bar_callback == progress_bar

    trainer.fit(model)


@pytest.mark.parametrize("lr_scheduler", [False, True])
def test_progress_bar_update_lr(tmp_path, lr_scheduler):
    config = deepcopy(MINIMAL_TRAINER_CFG)
    if lr_scheduler:
        config['trainer']['lr_scheduler'] = dict(
            method='StepLR', args=dict(step_size=1)
        )
    model = prepare_model(config, model_args=dict(num_data=10))

    class CustomTestProgressBar(VortexProgressBar):
        def _check_lr(self):
            lr_val = 0.0
            if len(self.trainer.lr_schedulers) > 0:
                lr_val = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            else:
                lr_val = self.trainer.optimizers[0].param_groups[0]["lr"]
            assert self.metrics_bar._fields['lr'] == lr_val

        def on_fit_start(self, trainer, pl_module):
            super().on_fit_start(trainer, pl_module)
            assert self.metrics_bar is None

        def on_train_start(self, trainer, pl_module):
            super().on_train_start(trainer, pl_module)
            self._check_lr()

        def on_epoch_start(self, trainer, pl_module):
            super().on_epoch_start(trainer, pl_module)
            self._check_lr()

    progress_bar = CustomTestProgressBar("TRAIN")
    trainer_args = dict(
        max_epochs=5,
        progress_bar_refresh_rate=101,   ## should not affect the provided pbar
    )
    trainer = patched_pl_trainer(str(tmp_path), model, callbacks=[progress_bar], trainer_args=trainer_args)
    assert trainer.progress_bar_callback == progress_bar

    trainer.fit(model)
