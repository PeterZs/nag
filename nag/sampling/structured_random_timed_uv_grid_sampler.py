from nag.sampling.random_timed_uv_grid_sampler import RandomTimedUVGridSampler
from enum import Enum
from typing import Optional, Tuple, Union
import torch
from nag.config.nag_config import NAGConfig
from nag.sampling.random_uv_grid_sampler import RandomUVGridSampler
from nag.sampling.timed_weighted_uv_grid_sampler import TimedWeightedUVGridSampler
from tools.util.torch import tensorify, index_of_first
from tools.util.typing import _DEFAULT, DEFAULT
from tools.logger.logging import logger
from tools.util.format import parse_enum
from nag.sampling.random_uv_grid_sampler import sample_uv_grid
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims


class StructureMode(Enum):
    """Structure mode for the structured random timed uv grid sampler."""

    NONE = "none"
    """No structure, just spatially random."""

    NEIGHBOR = "neighbor"
    """Spatially random sampling, but the following sample is a neighbor (= distance 1) of a random angle."""


class StructuredRandomTimedUVGridSampler(RandomTimedUVGridSampler):
    """Structured random timed uv grid sampler."""

    mode: StructureMode
    """Structure mode for the structured random timed uv grid sampler."""

    def __init__(self,
                 resolution: Tuple[int, int],
                 uv_max: Tuple[int, int],
                 t: torch.Tensor,
                 config: NAGConfig,
                 num_rays: Union[_DEFAULT, int] = DEFAULT,
                 num_batches: Union[_DEFAULT, int] = DEFAULT,
                 num_timestamps: Union[_DEFAULT, int] = DEFAULT,
                 deterministic: bool = False,
                 fixed_t: Optional[torch.Tensor] = None,
                 mode: Union[StructureMode, str] = StructureMode.NONE,
                 **kwargs
                 ) -> None:
        super().__init__(resolution=resolution, uv_max=uv_max, t=t, config=config, num_rays=num_rays,
                         num_batches=num_batches, num_timestamps=num_timestamps, deterministic=deterministic, fixed_t=fixed_t, **kwargs)
        self._mode = parse_enum(StructureMode, mode)

    @property
    def mode(self) -> StructureMode:
        """Get the structure mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: Union[StructureMode, str]):
        """Set the structure mode."""
        self._mode = parse_enum(StructureMode, mode)

    def sample_neighbor_grid(self, batch_idx: int, flatten: bool = True, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_sorted, t_weight = self._compute_ts_and_weights(
            batch_idx, flatten=flatten)

        seed = None if self._batch_seeds is None else self._batch_seeds[batch_idx].item(
        )

        sample_rays = self._num_rays // 2
        uv = sample_uv_grid(num_rays=sample_rays,
                            uv_max=self.uv_max,
                            dtype=self.config.dtype,
                            seed=seed
                            )
        if t_sorted is None:
            t_sorted = self._t

        if flatten:
            uv, _ = flatten_batch_dims(uv, -2)

        uv = uv.unsqueeze(1).repeat(1, 2, 1)  # Shape (sample_rays, 2, 2)
        angle = (torch.rand(sample_rays, dtype=self.config.dtype)
                 * 2 * torch.pi) - torch.pi  # Shape (sample_rays,)
        # X and Y offset for the neighbor. Shape (sample_rays, 2)
        offset = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        uv[:, 1:] = uv[:, 1:] + offset.unsqueeze(1)

        # Flatten the UV coordinates
        uv = uv.reshape(-1, 2)  # Shape (num_rays, 2)
        return uv, t_sorted, t_weight

    def sample_grid(
            self,
            batch_idx: int,
            flatten: bool = True,
            t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == StructureMode.NONE:
            return super().sample_grid(batch_idx, flatten=flatten, t=t)
        elif self.mode == StructureMode.NEIGHBOR:
            return self.sample_neighbor_grid(batch_idx, flatten=flatten, t=t)
        else:
            raise ValueError(f"Structure mode {self.mode} not supported.")
