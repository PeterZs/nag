from typing import Optional, Tuple, Union
from tools.metric.torch.metric import Metric
from tools.metric.torch.reducible import Reducible
import torch
from tools.metric.torch.module_mixin import ModuleMixin

class MaskLoss(Reducible):

    def __init__(self,
                 dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 reduction: str = "mean",
                 **kwargs) -> None:
        super().__init__(dim=dim, reduction=reduction, **kwargs)

    def __call__(self,
                 source: torch.Tensor,
                 target: torch.Tensor,
                 **kwargs
                 ) -> torch.Tensor:
        # Source and target are of shape (B, T, C)
        l1 = torch.abs((source - target.to(source.device, dtype=source.dtype)))
        return self.reduce(l1)
