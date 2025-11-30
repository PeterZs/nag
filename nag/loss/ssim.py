import skimage
from tools.metric.torch.metric import Metric
from torch import Tensor
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
import torch


class SSIM(Metric):

    def __init__(self,
                 name=None,
                 data_range=1.,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.data_range = data_range

    def __call__(self, source: Tensor, target: Tensor) -> Tensor:
        source, bs = flatten_batch_dims(source, -4)
        target, _ = flatten_batch_dims(target, -4)
        # Convert to numpy
        B, C, H, W = source.shape
        snp = source.detach().cpu().numpy()
        tnp = target.detach().cpu().numpy()
        # Compute SSIM
        res = torch.zeros(B, dtype=source.dtype, device=source.device)
        for i in range(B):
            try:
                res[i] = skimage.metrics.structural_similarity(snp[i], tnp[i],
                                                               data_range=self.data_range,
                                                               channel_axis=0).item()
            except ValueError as e:
                # Handle the case where the image is too small for SSIM calculation
                message = "win_size exceeds image extent"
                if message in str(e):
                    res[i] = torch.nan
                else:
                    raise e
        return unflatten_batch_dims(res, bs)
