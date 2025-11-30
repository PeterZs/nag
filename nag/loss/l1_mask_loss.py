from typing import Optional, Tuple, Union, Dict, Any
from tools.metric.torch.metric import Metric
from tools.metric.torch.reducible import Reducible
import torch
from nag.loss.l1 import L1
from nag.loss.mask_loss import MaskLoss
from tools.util.torch import tensorify, index_of_first
from nag.loss.mask_loss_mixin import MaskLossMixin


class L1MaskLoss(Reducible, MaskLossMixin):

    def __init__(self,
                 dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 reduction: str = "mean",
                 mask_loss_weight: float = 0.1,
                 **kwargs) -> None:
        super().__init__(dim=dim, reduction=reduction, **kwargs)
        self.l1 = L1(dim=dim, reduction=reduction, **kwargs)
        self.mask_loss = MaskLoss(dim=dim, reduction=reduction, **kwargs)
        self.mask_loss_weight = tensorify(mask_loss_weight).item()

    def __call__(self,
                 source: torch.Tensor,
                 target: torch.Tensor,
                 time_weight: Optional[torch.Tensor] = None,
                 context: Optional[torch.Tensor] = None,
                 metrics: Optional[Dict[str, Any]] = None,
                 **kwargs
                 ) -> torch.Tensor:
        # Source and target are of shape (B, T, 3)
        l1_value = self.l1(source, target, time_weight=time_weight)

        target_masks = context.get("masks")
        target_masks_ids = context.get("masks_object_idx").squeeze(0)

        alphas = context.get("object_alpha")
        alphas_ids = context.get("object_alpha_index")

        if len(alphas_ids) > 0:
            _map = index_of_first(target_masks_ids, alphas_ids)
            _filter = _map != -1
            exist_target_idx = _map[_filter]
            target_mask_compare = target_masks[..., exist_target_idx]
            alphas_filtered = alphas[..., _filter]

            mask_loss = self.mask_loss(alphas_filtered, target_mask_compare)
        else:
            mask_loss = torch.zeros_like(l1_value)

        if metrics is not None:
            metrics["loss/L1"] = l1_value.detach().cpu()
            metrics["loss/mask_L1"] = mask_loss.detach().cpu()

        return l1_value + self.mask_loss_weight * mask_loss
