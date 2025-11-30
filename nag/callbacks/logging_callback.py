from typing import Optional, Union
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from tools.logger.logging import logger
from logging import Logger
from tools.util.typing import DEFAULT, _DEFAULT
from tools.error.argument_none_error import ArgumentNoneError

class LoggingCallback(ProgressBar):

    logger: Logger
    """Logger to use. Default is the default logger."""

    log_steps: bool
    """Whether to log steps. Default is False."""

    log_epoch: bool
    """Whether to log epoch. Default is True."""

    _epoch_format: Optional[str]

    def __init__(self, 
                 log_steps: bool = False,
                 log_epoch: bool = True,
                 logger: Union[Logger, _DEFAULT] = DEFAULT):
        super().__init__()
        if logger is None:
            raise ArgumentNoneError("logger")
        if logger == DEFAULT:
            from tools.logger.logging import logger as default_logger
            self.logger = default_logger
        else:
            self.logger = logger
        self.log_steps = log_steps
        self.log_epoch = log_epoch
        self._epoch_format = None

    def get_epoch_format(self, max_epochs: int) -> str:
        if self._epoch_format is None:
            digits = len(str(max_epochs))
            self._epoch_format = f"{{:{digits}d}}"
        return self._epoch_format

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logger.info("Training started.")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logger.info("Training ended.")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        fmt = self.get_epoch_format(trainer.max_epochs)
        self.logger.info(f"Epoch ({fmt.format(trainer.current_epoch)} / {trainer.max_epochs}) started.")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        fmt = self.get_epoch_format(trainer.max_epochs)
        metrics = self.get_metrics(trainer, pl_module)
        metric_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"Epoch ({fmt.format(trainer.current_epoch)} / {trainer.max_epochs}) ended. {metric_str}")

