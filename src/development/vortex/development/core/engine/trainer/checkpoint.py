from typing import Optional, Union
from pathlib import Path

from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector as PLCheckpointConnector
from pytorch_lightning.callbacks import ModelCheckpoint as PLModelCheckpoint


class CheckpointConnector(PLCheckpointConnector):
    """patch for additional data in checkpoint
    """
    def __init__(self, trainer):
        super().__init__(trainer)

    def dump_checkpoint(self, weights_only: bool) -> dict:
        checkpoint = super().dump_checkpoint(weights_only=weights_only)
        checkpoint['vortex_version'] = "0.3.0+dev-sdfas798"
        checkpoint['metrics'] = self.trainer.logger_connector.callback_metrics

        model = self.trainer.get_model()
        checkpoint['class_names'] = model.class_names
        checkpoint['config'] = model.config
        return checkpoint


class ModelCheckpoint(PLModelCheckpoint):
    """Another way to apply custom behavior, but this will be version dependent
    """
    FILE_EXTENSION = ".pth"

    def __init__(
        self,
        filepath: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: Optional[int] = None,
        save_weights_only: bool = False,
        mode: str = "auto",
        period: int = 1,
        prefix: str = "",
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
    ):
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode, period=period,
            prefix=prefix, dirpath=dirpath,
            filename=filename
        )