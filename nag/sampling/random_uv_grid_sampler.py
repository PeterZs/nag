import math
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import torch
from torch.utils.data import Dataset

from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.epoch_state_mixin import EpochStateMixin
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from tools.util.format import raise_on_none
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
from torch.nn import functional as F
from tools.util.torch import tensorify

from nag.sampling.uv_grid_sampler import UVGridSampler
from tools.logger.logging import logger


def sample_uv_grid(
    num_rays: int,
    uv_max: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    distribution: Literal["uniform", "normal", "pixel"] = "uniform"
) -> torch.Tensor:
    """Samples pixel grid for the camera image plane.

    Parameters
    ----------

    num_rays : int
        Number of rays to sample for the grid.

    uv_max : Optional[torch.Tensor], optional
        Maximum UV coordinate. The UV coordinates will be scaled to this value [0, uv_max), by default None
        Shape is (2,) where 2 is the (x, y) coordinate.

    dtype : torch.dtype, optional
        Data type for the tensor, by default torch.float32

    seed : Optional[int], optional
        Seed for the random number generator, by default None

    Returns
    -------
    torch.Tensor
        The dense rays as (1, num_rays, 2) of (x, y) tensor.
    """
    # Create a grid of pixel coordinates

    if seed is not None:
        torch.manual_seed(seed)
    if distribution == "uniform":
        uv = torch.rand((num_rays, 2), dtype=dtype)
        uv *= uv_max
    elif distribution == "normal":
        # Sample from a random uniform distribution to get a pixel position,
        # Add inter pixel noise from a normal distribution
        uv = torch.empty((num_rays, 2), dtype=dtype)
        uv[:, 0] = torch.randint(0, uv_max[0], (num_rays,), dtype=dtype)
        uv[:, 1] = torch.randint(0, uv_max[1], (num_rays,), dtype=dtype)
        uv = uv + torch.randn((num_rays, 2), dtype=dtype) * 0.2
    elif distribution == "pixel":
        # Sample from a random uniform distribution to get a pixel position,
        # Add inter pixel noise from a normal distribution
        uv = torch.empty((num_rays, 2), dtype=dtype)
        uv[:, 0] = torch.randint(0, uv_max[0], (num_rays,), dtype=dtype)
        uv[:, 1] = torch.randint(0, uv_max[1], (num_rays,), dtype=dtype)
    return uv


class RandomUVGridSampler(UVGridSampler, EpochStateMixin):

    _config: NAGConfig
    """Configuration for the NAG model."""

    _num_rays: int
    """Number of rays to sample for the grid."""

    _batch_seeds: Optional[torch.Tensor]
    """Seeds for the random number generator for each batch."""

    _num_batches: int
    """Number of batches to sample for the grid."""

    _num_rays: int
    """Number of rays to sample for the grid."""

    _epoch_seeds: Optional[torch.Tensor]
    """Seeds for the random number generator for each epoch."""

    _deterministic: bool
    """Deterministic sampling."""

    def __init__(self,
                 num_rays: int,
                 num_batches: int,
                 resolution: Tuple[int, int],
                 uv_max: Tuple[int, int],
                 t: torch.Tensor,
                 config: NAGConfig,
                 deterministic: bool = False,
                 distribution: Literal["uniform", "normal"] = "uniform",
                 ) -> None:
        super().__init__(resolution=resolution, uv_max=uv_max, t=t, dtype=config.dtype)
        self._config = raise_on_none(config)
        self._num_rays = raise_on_none(num_rays)
        self._num_batches = raise_on_none(num_batches)
        self._deterministic = deterministic
        self._distribution = distribution
        self.set_epoch_seeds(config.max_epochs)
        self.set_batch_seeds(0)

    def set_epoch_seeds(self, num_epochs: int) -> None:
        if self._deterministic:
            self._epoch_seeds = torch.randint(0, 2**32, (num_epochs,))
        else:
            self._epoch_seeds = None

    def set_batch_seeds(self, epoch_idx: int) -> None:
        if self._deterministic:
            if self._epoch_seeds is not None:
                torch.manual_seed(self._epoch_seeds[epoch_idx].item())
            self._batch_seeds = torch.randint(0, 2**32, (self._num_batches,))
        else:
            self._batch_seeds = None

    def on_epoch_change(self, epoch: int) -> None:
        self.set_batch_seeds(epoch)

    @property
    def config(self) -> NAGConfig:
        return self._config

    def __len__(self) -> int:
        return self._num_batches

    def sample_grid(
            self,
            batch_idx: int,
            flatten: bool = True,
            t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # UV coordinates
        seed = None if self._batch_seeds is None else self._batch_seeds[batch_idx].item(
        )
        uv = sample_uv_grid(num_rays=self._num_rays,
                            uv_max=self.uv_max,
                            dtype=self.config.dtype,
                            seed=seed,
                            distribution=self._distribution
                            )
        if t is None:
            t = self._t

        if flatten:
            uv, _ = flatten_batch_dims(uv, -2)
        return uv, t

    def reassemble_grid(self,
                        data: torch.Tensor,
                        batch_idx: Optional[int] = None,
                        uv: Optional[torch.Tensor] = None,
                        return_hit_mask: bool = False
                        ) -> torch.Tensor:
        if uv is None:
            if batch_idx is None:
                raise ValueError("Either batch_idx or uv must be provided.")
            else:
                if self._batch_seeds is None:
                    logger.warning(
                        "The dataloader is non-deterministic. The result will be wrong, provide uv instead.")
                uv, _ = self.sample_grid(batch_idx, flatten=True)
        if len(uv.shape) == 2:
            uv = uv.unsqueeze(1)
        B, T, _ = uv.shape
        data, data_shape = flatten_batch_dims(data, -3)
        D, B, T = data.shape
        rounded_uv = torch.round(uv).int().permute(1, 0, 2)  # T, B, 2
        unfolded_data = torch.zeros(
            (T, ) + tuple(self._uv_max) + (D, ), dtype=self.config.dtype)

        sel_x = torch.clamp(rounded_uv[..., 0], 0, self._uv_max[0] - 1)
        sel_y = torch.clamp(rounded_uv[..., 1], 0, self._uv_max[1] - 1)
        unfolded_data[torch.arange(T), sel_x, sel_y] = data.permute(
            2, 1, 0)  # T, X, Y, D
        unfolded_data = unfolded_data.permute(3, 0, 2, 1)  # D, T, Y, X
        unfolded_data_org = unflatten_batch_dims(unfolded_data, data_shape)
        if not return_hit_mask:
            return unfolded_data_org
        hit_mask = torch.zeros((T, ) + tuple(self._uv_max), dtype=torch.bool)
        hit_mask[torch.arange(T), sel_x, sel_y] = True
        return unfolded_data_org, hit_mask
