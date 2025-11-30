from typing import Type
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.amp.grad_scaler import GradScaler
from tools.util.format import raise_on_none


class MixedPrecisionCustom(MixedPrecision):
    def __init__(self,
                 precision: str,
                 device: str,
                 scaler_type: Type[GradScaler],
                 scaler_kwargs: dict = None
                 ):
        super().__init__(precision, device, None)
        raise_on_none(scaler_type)
        if self.precision == "16-mixed":
            args = scaler_kwargs or {}
            scaler = scaler_type(**args)
        else:
            raise ValueError(f"`Passed `{type(self).__name__}(precision={precision!r})`."
                             f" Precision must be '16-mixed' for using {type(self).__name__}.")
        self.scaler = scaler
