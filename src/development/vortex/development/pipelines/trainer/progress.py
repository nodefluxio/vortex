import sys
import enlighten

from pytorch_lightning.callbacks.progress import ProgressBarBase, convert_inf


class VortexProgressBar(ProgressBarBase):
    def __init__(self, stage: str, refresh_rate: int = 1, process_position: int = 0, show_lr: bool = True):
        super().__init__()
        self._refresh_rate = refresh_rate
        self._process_position = process_position
        self._enabled = True
        self._stage = stage
        self._current_epoch = 0
    
        self.manager = None
        self.status_bar = None
        self.main_progress_bar = None
        self.train_progress_bar = None
        self.val_progress_bar = None
        self.metrics_bar = None
        self.test_progress_bar = None

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state['manager'] = None
        state['main_progress_bar'] = None
        state['train_progress_bar'] = None
        state['val_progress_bar'] = None
        state['test_progress_bar'] = None
        return state

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def init_manager(self, enabled=True) -> enlighten.Manager:
        self.manager = enlighten.get_manager(stream=sys.stdout, enabled=enabled)
        self.status_bar = self.manager.status_bar(
            status_format=u'Vortex{fill}{stage}{fill}{vnum}  {elapsed}',
            color='bold_underline_white', justify=enlighten.Justify.CENTER, 
            autorefresh=True, min_delta=0.5, stage=self._stage, 
            vnum="", position=(5*self.process_position + 5)
        )

    def init_metrics_bar(self) -> enlighten.StatusBar:
        lr_format = "" if self.trainer.testing else "lr: {lr:.4g}  "
        kwargs = dict() if self.trainer.testing else dict(lr=0.0)
        metric_format = "    " + lr_format + "{metrics}{fill}"

        metric_bar = self.manager.status_bar(
            status_format=metric_format,
            position=(5*self.process_position + 1),
            justify=enlighten.Justify.LEFT,
            metrics="", **kwargs
        )
        self.manager.status_bar(
            status_format="{fill}",
            position=(5*self.process_position + 2)
        )
        return metric_bar

    def init_epoch_pbar(self, total=None) -> enlighten.Counter:
        bar = self.manager.counter(
            desc="Epoch:",
            position=(5*self.process_position + 4),
            leave=True,
            enabled=self.is_enabled,
            total=total,
            unit="epoch"
        )
        return bar

    def init_sanity_pbar(self, total) -> enlighten.Counter:
        bar = self.manager.counter(
            desc="Validation sanity check:",
            position=(5*self.process_position + 4),
            leave=False, enabled=False,
            total=total
        )
        return bar

    def init_train_pbar(self, total) -> enlighten.Counter:
        bar = self.manager.counter(
            desc="  Training:",
            total=total,
            count=self.train_batch_idx,
            position=(5*self.process_position + 3),
            enabled=self.is_enabled,
            leave=False,
            unit="it"
        )
        return bar

    def init_validation_pbar(self, total) -> enlighten.Counter:
        bar = self.manager.counter(
            desc="  Validating:",
            total=total,
            count=self.val_batch_idx,
            position=(5*self.process_position + 3),
            enabled=self.is_enabled,
            leave=False,
            unit="it"
        )
        return bar

    def init_test_pbar(self, total) -> enlighten.Counter:
        bar = self.manager.counter(
            desc="  Testing:",
            total=total,
            count=self.test_batch_idx,
            position=(5*self.process_position + 4),
            enabled=self.is_enabled,
            leave=True,
            unit="it"
        )
        return bar

    def refresh_lr(self):
        assert self.metrics_bar is not None, "initiate 'metrics_bar' first "\
            "before calling 'refresh_lr'"
        if self.trainer.testing:
            return

        if len(self.trainer.lr_schedulers) > 0:
            lr_val = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        else:
            lr_val = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.metrics_bar.update(lr=lr_val)

    def refresh_metrics(self):
        assert self.metrics_bar is not None, "initiate 'metrics_bar' first "\
            "before calling 'refresh_metrics'"
        if not self.trainer.testing:
            self.refresh_lr()

        metrics_str = ""
        for name, val in self.trainer.progress_bar_metrics.items():
            metrics_str += "{}: {:.4g}  ".format(name, val)
        self.metrics_bar.update(metrics=metrics_str)

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self.init_manager()
        vnum_str = "version: {}".format(trainer.progress_bar_dict["v_num"])
        self.status_bar.update(vnum=vnum_str)

    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)
        self.status_bar.close()
        self.manager.stop()

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        total = convert_inf(sum(trainer.num_sanity_val_batches))
        self.val_progress_bar = self.init_sanity_pbar(total)

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        self.val_progress_bar.close()

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = self.init_epoch_pbar(trainer.max_epochs)
        self.main_progress_bar.refresh()
        self.metrics_bar = self.init_metrics_bar()
        self.refresh_lr()

        self._current_epoch = trainer.current_epoch
        self.main_progress_bar.count = trainer.current_epoch


    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        total = convert_inf(self.total_train_batches)
        self.train_progress_bar = self.init_train_pbar(total)
        self.train_progress_bar.refresh()
        self.refresh_lr()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
            self.refresh_metrics()
            self.train_progress_bar.update(self.refresh_rate)
        if self.is_enabled and self.train_batch_idx >= self.train_progress_bar.total:
            self.train_progress_bar.close()

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if not trainer.running_sanity_check:
            total = convert_inf(self.total_val_batches)
            self.val_progress_bar = self.init_validation_pbar(total)
            self.val_progress_bar.refresh()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.val_batch_idx % self.refresh_rate == 0:
            if not self.trainer.running_sanity_check:
                self.refresh_metrics()
            self.val_progress_bar.update(self.refresh_rate)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if not trainer.running_sanity_check:
            self.refresh_metrics()
            self.val_progress_bar.close()
            self.main_progress_bar.update()

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.refresh_metrics()
        self.main_progress_bar.close()

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        total = convert_inf(self.total_test_batches)
        self.test_progress_bar = self.init_test_pbar(total)
        self.test_progress_bar.refresh()
        self.metrics_bar = self.init_metrics_bar()
        self.refresh_lr()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.test_batch_idx % self.refresh_rate == 0:
            self.refresh_metrics()
            self.test_progress_bar.update(self.refresh_rate)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        self.refresh_metrics()
        self.test_progress_bar.close()
        self.manager.stop()
