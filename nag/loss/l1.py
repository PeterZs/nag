from typing import Optional, Tuple, Union
from tools.metric.torch.metric import Metric
from tools.metric.torch.reducible import Reducible
import torch


class L1(Reducible):

    def __init__(self,
                 dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 reduction: str = "mean",
                 **kwargs) -> None:
        super().__init__(dim=dim, reduction=reduction, **kwargs)

    def __call__(self,
                 source: torch.Tensor,
                 target: torch.Tensor,
                 time_weight: Optional[torch.Tensor] = None,
                 **kwargs
                 ) -> torch.Tensor:
        # Source and target are of shape (B, T, 3)
        l1 = torch.abs((source - target))
        if time_weight is not None:
            l1 = l1 * \
                time_weight.unsqueeze(
                    0).unsqueeze(-1).repeat(l1.shape[0], 1, l1.shape[-1])
        l1_loss = self.reduce(l1)
        return l1_loss
