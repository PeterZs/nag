from typing import Optional, Tuple, Union
from tools.metric.torch.metric import Metric
from tools.metric.torch.reducible import Reducible
import torch

class WL1(Reducible):

    def __init__(self, 
                 dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 reduction: str = "mean",
                 eps: float = 0.001,
                 **kwargs) -> None:
        super().__init__(dim=dim, reduction=reduction, **kwargs)
        self.eps = eps


    def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        wl1 = (torch.abs((source - target) /
                             (source.detach() + self.eps)))
        return self.reduce(wl1)