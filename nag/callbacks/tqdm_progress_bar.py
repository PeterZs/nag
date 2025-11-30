from typing import Any, Mapping, Optional, Union
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar as OriginalTQDMProgressBar
from torch import Tensor


class TQDMProgressBar(OriginalTQDMProgressBar):

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Optional[Union[Tensor, Mapping[str, Any]]], batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if outputs is not None and isinstance(outputs, dict) and "progress_bar" in outputs:
            self.train_progress_bar.set_postfix(outputs["progress_bar"])