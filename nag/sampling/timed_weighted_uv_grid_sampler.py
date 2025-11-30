import abc
from typing import Optional, Tuple
import torch

from nag.sampling.uv_grid_sampler import UVGridSampler


class TimedWeightedUVGridSampler(UVGridSampler):

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample_grid(idx, flatten=self._flatten, t=self._t)

    @abc.abstractmethod
    def sample_grid(
            self,
            batch_idx: int,
            flatten: bool = True,
            t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample grid for the given batch index.

        Parameters
        ----------

        batch_idx : int
            Index of the batch to sample rays for.
            Each batch contains a spatially subsampled set of rays defined by resolution.
            The value range of the UV coordinates is dependent on uv_max.

        flatten : bool, optional
            Whether to flatten the batch dimensions, by default True
            If false, the batch dimensions are kept. (H, W, 2) instead of (H * W, 2)

        t : Optional[torch.Tensor], optional
            Timestamps to get the object position for, by default None
            If none, the timestamps for the camera are used. Should be of shape (T,) in range [0, 1]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            1. Sample grid
            The UV / Grid coordinates of the rays. Shape: (B, 2)
            (x, y) coordinates of the rays in the image.
            While the x, y are in range [0, uv_max[0]], [0, uv_max[1]] respectively.

            2. Times
            The sample times at the time dimension. Shape: (T,)

            3. Timed Weights
            The weights for the time dimension. Shape: (T,)

        """
        raise NotImplementedError()
