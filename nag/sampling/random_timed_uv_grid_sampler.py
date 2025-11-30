

from typing import Optional, Tuple, Union
import torch
from nag.config.nag_config import NAGConfig
from nag.sampling.random_uv_grid_sampler import RandomUVGridSampler
from nag.sampling.timed_weighted_uv_grid_sampler import TimedWeightedUVGridSampler
from tools.util.torch import tensorify, index_of_first
from tools.util.typing import _DEFAULT, DEFAULT
from tools.logger.logging import logger


class RandomTimedUVGridSampler(
        RandomUVGridSampler,
        TimedWeightedUVGridSampler):

    _fixed_t: Optional[torch.Tensor]
    """Fixed timestamps which need to be evaluated for the ray positions. Shape is (T,) where T is the number of timestamps."""

    _num_timestamps: int
    """Number of timestamps to sample for the grid within each batch."""

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
                 **kwargs
                 ) -> None:
        num_rays, num_batches, num_timestamps = self.calculate_specs(
            config, t, resolution, num_rays=num_rays, num_batches=num_batches, num_timestamps=num_timestamps)
        super().__init__(num_rays=num_rays, num_batches=num_batches, resolution=resolution,
                         uv_max=uv_max, t=t, config=config, deterministic=deterministic, **kwargs)
        self._fixed_t = tensorify(fixed_t) if fixed_t is not None else None
        self._num_timestamps = num_timestamps

    def calculate_specs(self,
                        config: NAGConfig,
                        t: torch.Tensor,
                        resolution: Tuple[int, int],
                        num_rays: Union[_DEFAULT, int] = DEFAULT,
                        num_batches: Union[_DEFAULT, int] = DEFAULT,
                        num_timestamps: Union[_DEFAULT, int] = DEFAULT,
                        ) -> Tuple[int, int, int]:
        # Calculate the number of rays, batches and timestamps, if there are not set.
        msg = ""
        max_total_batch_size = config.max_total_batch_size
        num_frames = len(t)

        if num_timestamps != DEFAULT:
            if num_timestamps > len(t):
                logger.warning(
                    f"Number of timestamps is larger than the number of timestamps in t. Setting it to {len(t)}.")
                num_timestamps = len(t)

        unique_data_per_epoch = torch.tensor(resolution).prod() * num_frames
        data_per_epoch = unique_data_per_epoch * config.epoch_data_ratio

        if max_total_batch_size is None and ((num_rays == DEFAULT) or (num_batches == DEFAULT) or (num_timestamps == DEFAULT)):
            # Set the maximum total batch size to 20. M
            max_total_batch_size = int(2e7)
            msg = f"\nMaximum total batch size is not set. Setting it to {max_total_batch_size}."

        num_defaults = (num_rays == DEFAULT) + (num_batches ==
                                                DEFAULT) + (num_timestamps == DEFAULT)
        times_was_lower = 1.

        while num_defaults > 1:
            if num_timestamps == DEFAULT:
                num_timestamps = min(20, len(t))
                times_was_lower = 20 / num_timestamps
                msg += f"\nNumber of timestamps is not set. Setting it to {num_timestamps}."
                num_defaults -= 1
                continue
            if num_rays == DEFAULT:
                num_rays = int(1.4e5) * times_was_lower
                msg += f"\nNumber of rays is not set. Setting it to {num_rays}."
                num_defaults -= 1
                continue

        if num_timestamps == DEFAULT:
            # Calculate the number of timestamps based on the maximum total batch size
            num_timestamps = data_per_epoch // (num_rays * num_batches)
            msg += f"\nNumber of timestamps is not set. Calculating it to {num_timestamps}."
        if num_rays == DEFAULT:
            # Calculate the number of rays based on the maximum total batch size
            num_rays = data_per_epoch // (num_timestamps * num_batches)
            msg += f"\nNumber of rays is not set. Calculating it to {num_rays}."
        if num_batches == DEFAULT:
            # Calculate the number of batches based on the maximum total batch size
            num_batches = max(torch.ceil(
                data_per_epoch / (num_timestamps * num_rays)), 1).int()
            msg += f"\nNumber of batches is not set. Calculating it to {num_batches}."
        # Check if the total batch size is within the maximum total batch size
        total_batch_size = num_rays * num_timestamps
        if total_batch_size > max_total_batch_size:
            # Reduce the rays, to fit the maximum total batch size
            new_num_rays = max_total_batch_size // num_timestamps
            # Increase the number of batches to fit the maximum total batch size
            new_num_batches = max(torch.ceil(
                data_per_epoch / (num_timestamps * num_rays)), 1).int()
            if new_num_batches != num_batches:
                msg += f"\nTotal batch size is larger than the maximum total batch size. Reducing the number of rays from {num_rays} to {new_num_rays} and increasing the number of batches from {num_batches} to {num_batches}."
            else:
                msg += f"\nTotal batch size is larger than the maximum total batch size. Reducing the number of rays from {num_rays} to {new_num_rays}."
            num_rays = new_num_rays
            num_batches = new_num_batches
        if len(msg) > 0:
            msg += f"\nCovering {data_per_epoch} datapoints (Data/Epoch Ratio: {config.epoch_data_ratio}) per epoch by splitting in: \nNumber of rays: {num_rays}, Number of batches: {num_batches}, Number of timestamps: {num_timestamps}. "
            logger.info(msg.strip("\n"))
        num_rays = int(num_rays)
        num_batches = int(num_batches)
        num_timestamps = int(num_timestamps)
        return num_rays, num_batches, num_timestamps

    @property
    def fixed_t(self) -> Optional[torch.Tensor]:
        return self._fixed_t

    @fixed_t.setter
    def fixed_t(self, value: Optional[torch.Tensor]) -> None:
        if value is not None:
            value = tensorify(value, dtype=self.dtype)
            value = torch.unique(value, sorted=True, dim=0)
            # Check if all are in t
            idf = index_of_first(self._t, value)
            if (idf < 0).any():
                raise ValueError(
                    f"Fixed timestamps should be within the timestamps for the camera. {value[(idf < 0)]} not in t")
        self._fixed_t = value
        if self._fixed_t is not None:
            if self.num_timestamps <= len(self._fixed_t):
                if len(self._fixed_t) < len(self._t):
                    raise ValueError(f"Number of timestamps for each batch is smaller than the number of fixed timestamps. Learning of scene is not possible as there is no room for learning other timestamps. Increase num_timestamps or decrease the number of fixed timestamps.")

    @property
    def num_timestamps(self) -> int:
        return self._num_timestamps

    def _compute_ts_and_weights(self,
                                batch_idx: int,
                                flatten: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        fixed_t = self._fixed_t if self._fixed_t is not None else torch.zeros(
            0, dtype=self.dtype)
        n_fixed_t = len(fixed_t)
        # Randomly sample the timestamps for the batch
        sample_count = self.num_timestamps - n_fixed_t

        if sample_count == 0:
            # Edge case where no timestamps are sampled as fixed timestamps cover all timestamps in t
            return super().sample_grid(batch_idx, flatten=flatten, t=fixed_t), torch.ones(n_fixed_t, dtype=self.dtype)

        if sample_count > len(self._t):
            # Sample the timestamps from the t
            sampled_idx = torch.randint(
                0, len(self._t), (sample_count, ), dtype=torch.int32)

        else:
            sampled_idx = torch.multinomial(torch.ones_like(
                self._t) / len(self._t), sample_count, replacement=False)  # Draw unique samples from the t

        sampled_t = self._t[sampled_idx]

        # Remove fixed ts if they are in the sampled ts
        fixed_t = fixed_t[index_of_first(sampled_t, fixed_t) < 0]

        unique_t, counts = torch.unique(
            sampled_t, sorted=True, return_counts=True, dim=0)
        t_weight = torch.zeros(len(unique_t) + len(fixed_t), dtype=torch.int32)
        t_weight[:len(unique_t)] = counts

        assembled_t = torch.cat([unique_t, fixed_t], dim=0)
        t_sorted_idx = torch.argsort(assembled_t, dim=0)
        t_sorted = assembled_t[t_sorted_idx]
        t_weight = t_weight[t_sorted_idx]
        return t_sorted, t_weight

    def sample_grid(
            self,
            batch_idx: int,
            flatten: bool = True,
            t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_sorted, t_weight = self._compute_ts_and_weights(
            batch_idx, flatten=flatten)
        return *super().sample_grid(batch_idx, flatten=flatten, t=t_sorted), t_weight
