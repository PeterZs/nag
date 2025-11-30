import math
from typing import Any, Callable, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import torch
from torch.utils.data import Dataset

from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from tools.util.format import raise_on_none
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
from torch.nn import functional as F
from tools.util.torch import tensorify

from nag.sampling.uv_grid_sampler import UVGridSampler


def sample_uv_grid(
    resolution: torch.Tensor,
    subsample: Union[int, torch.Tensor] = 1,
    subsample_offset: Union[int, torch.Tensor] = 0,
    align_corners: bool = True,
    uv_max: Optional[torch.Tensor] = None,
    inter_pixel_noise_fnc: Optional[Callable[[
        torch.Size], torch.Tensor]] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Samples pixel grid for the camera image plane.

    Parameters
    ----------

    resolution : torch.Tensor
        Resolution of the image plane.
        Shape (2,) (x, y) (width, height) of the image plane to sample from.

    subsample : Union[int, torch.Tensor], optional
        Subsample factor for the rays, by default 1.
        If subsample is 1, the grid will be the same size as the image resolution.
        Can be a single integer or a tensor of shape (2,) (x, y) for the subsample factor.

    subsample_offset : Union[int, torch.Tensor], optional
        Offset for the subsample, by default 0
        Will start the subsample at the given offset.
        Can be a single integer or a tensor of shape (2,) (x, y) for the offset.

    align_corners : bool, optional
        If True, 0, 0  is the top left center of the pixel, by default True.
        If false, 0, 0 is the top left corner of the pixel.

    uv_max : Optional[torch.Tensor], optional
        Maximum UV coordinate. If provided, the UV coordinates will be scaled to this value, by default None

    inter_pixel_noise_fnc : Optional[Callable[[torch.Size], torch.Tensor]], optional
        Function to generate inter pixel noise for the grid, by default None
        Assures that a somewhat continouus coordinate space is sampled.
        torch.size is the shape of the grid usually (H, W, 2),
        the noise should be in the range [0, 1] and will be subtracted by 0.5.
        A uniform sampling => torch.rand(shape) is a good choice.

        Example:
        ```python
        def noise_fnc(shape: torch.Size) -> torch.Tensor:
            return torch.rand(shape)
        ```


    Returns
    -------
    torch.Tensor
        The dense rays as (height, width, 3) of (x, y, z) tensor with z = 0.
    """
    # Create a grid of pixel coordinates

    idx, idy = None, None
    if isinstance(subsample, int):
        subsample = torch.tensor(
            subsample, dtype=torch.int32).unsqueeze(0).repeat(2)
    if isinstance(subsample_offset, int):
        subsample_offset = torch.tensor(
            subsample_offset, dtype=torch.int32).unsqueeze(0).repeat(2)

    _sample_resolution = resolution

    if (subsample == 1).all():
        if align_corners:
            idx = torch.linspace(0, _sample_resolution[0],
                                 _sample_resolution[0], dtype=dtype, device=_sample_resolution.device)
            idy = torch.linspace(0, _sample_resolution[1],
                                 _sample_resolution[1], dtype=dtype, device=_sample_resolution.device)
        else:
            raise NotImplementedError()
            idx = torch.arange(subsample_offset[0], _sample_resolution[0], subsample[0],
                               dtype=dtype, device=_sample_resolution.device) + 0.5 / _sample_resolution[0]
            idy = torch.arange(subsample_offset[1], _sample_resolution[1], subsample[1],
                               dtype=dtype, device=_sample_resolution.device) + 0.5 / _sample_resolution[1]
    else:
        if align_corners:
            idx = torch.linspace(0, _sample_resolution[0], _sample_resolution[0], dtype=dtype, device=_sample_resolution.device)[
                subsample_offset[0]::subsample[0]]
            idy = torch.linspace(0, _sample_resolution[1], _sample_resolution[1], dtype=dtype, device=_sample_resolution.device)[
                subsample_offset[1]::subsample[1]]
        else:
            raise NotImplementedError()
            idx = torch.arange(subsample_offset[0], _sample_resolution[0], _sample_resolution[0] //
                               subsample[0], dtype=dtype, device=_sample_resolution.device) + 0.5 / _sample_resolution[0]
            idy = torch.arange(subsample_offset[1], _sample_resolution[1], _sample_resolution[1] //
                               subsample[1], dtype=dtype, device=_sample_resolution.device) + 0.5 / _sample_resolution[1]

    # Coordinate order is (x, y), (height, width)
    grid = torch.stack(torch.meshgrid(
        idx, idy, indexing="xy"), dim=-1).to(dtype)

    if inter_pixel_noise_fnc is not None:
        noise = inter_pixel_noise_fnc(grid.shape).to(
            dtype=dtype, device=_sample_resolution.device) - 0.5
        grid += noise

    if uv_max is not None:
        # Multiply by the ratio of the resolutions to query the overall image
        ratio = uv_max / _sample_resolution
        grid[..., 0] = grid[..., 0] * ratio[0]
        grid[..., 1] = grid[..., 1] * ratio[1]

    return grid


class RegularUVGridSampler(UVGridSampler):

    _subsample_offsets: torch.Tensor
    """Subsample offsets for the rays and grid pixels.
    This order assures that every ray is sampled at least once per batch.
    Shape is (N, 2) where N is the number of batches and 2 is the (x, y) offset for the subsample.
    """
    _subsample: torch.Tensor
    """Subsample factor (2, ) (x, y) for the rays and grid pixels. 1 means no subsampling."""

    _config: NAGConfig
    """Configuration for the NAG model."""

    _inter_pixel_noise_fnc: Optional[Callable[[torch.Size], torch.Tensor]]
    """Function to generate inter pixel noise for the grid, by default None"""

    def __init__(self,
                 resolution: Tuple[int, int],
                 uv_max: Tuple[int, int],
                 t: torch.Tensor,
                 config: NAGConfig,
                 inter_pixel_noise_fnc: Optional[Callable[[
                     torch.Size], torch.Tensor]] = None,
                 max_total_batch_size: Optional[int] = None,
                 align_corners: bool = True,
                 ) -> None:
        super().__init__(resolution=resolution, uv_max=uv_max, t=t, dtype=config.dtype)
        self._config = raise_on_none(config)
        self.max_total_batch_size = config.max_total_batch_size if max_total_batch_size is None else max_total_batch_size
        self._subsample, self._subsample_offsets = self.calculate_subsample_offsets(
            num_objects=config.num_objects,
            n_t=len(self._t),
            max_total_batch_size=self.max_total_batch_size,
            resolution=self._resolution,
            num_batches=self.config.num_batches)
        self._inter_pixel_noise_fnc = inter_pixel_noise_fnc
        self.align_corners = align_corners

    def calculate_subsample_offsets(self,
                                    num_objects: int, n_t: int,
                                    resolution: Tuple[int, int],
                                    max_total_batch_size: Optional[int] = None,
                                    num_batches: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the subsample offsets for the rays."""
        if num_batches is None:
            if max_total_batch_size is None:
                raise ValueError(
                    "Either num_batches or max_total_batch_size should be provided.")
            num_rays = int(math.floor(
                max_total_batch_size / n_t / num_objects))
            total_rays = resolution[0] * resolution[1]
            num_batches = int(math.ceil(total_rays / num_rays))
        bt = torch.tensor(num_batches)
        bt = bt.unsqueeze(0)
        bt = bt.ceil()
        bt = bt.int()
        bt = bt.repeat(2)
        subsample = bt
        # subsample = torch.sqrt(torch.tensor(num_batches)).unsqueeze(
        #    0).ceil().int().repeat(2)
        x = torch.arange(subsample[0])
        y = torch.arange(subsample[1])
        offsets = torch.stack(torch.meshgrid(
            x, y, indexing="xy"), dim=-1).reshape(-1, 2)
        return subsample, offsets

    @property
    def config(self) -> NAGConfig:
        return self._config

    @property
    def t(self) -> torch.Tensor:
        return self._t

    @t.setter
    def t(self, t: torch.Tensor) -> None:
        if t != self._t:
            self._subsample, self._subsample_offsets = self.calculate_subsample_offsets(
                num_objects=self.config.num_objects,
                n_t=len(t),
                max_total_batch_size=self.max_total_batch_size,
                resolution=self._resolution,
                num_batches=self.config.num_batches)
            self._t = t

    def __len__(self) -> int:
        return len(self._subsample_offsets)

    def batch_tensor_to_image_tensor(self, batch_tensor: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Convert the batch tensor to an image tensor for the given batch index.

        Parameters
        ----------

        batch_tensor : torch.Tensor
            Batch tensor to convert to an image tensor. Should be in Shape (B, T, C)

        batch_idx : int
            Index of the batch to convert to an image tensor. Is used to determine the shape of the image tensor (H, W).

        Returns
        -------

        torch.Tensor
            Image tensor in Shape (T, C, H, W)
        """
        H, W = self.get_batch_shape(batch_idx)
        bt = unflatten_batch_dims(batch_tensor, (H, W))
        # Move Time to front, Channel to 2nd
        bt = bt.permute(2, 3, 0, 1)
        return bt

    def get_possible_batch_shapes(self) -> List[Tuple[int, int]]:
        """Get the possible shapes of the batch."""
        W_raw, H_raw = self._resolution / self._subsample
        # If H_raw or W_raw is not an integer, we need to ceil / floor it
        if int(H_raw) == H_raw and int(W_raw) == W_raw:
            return (int(H_raw), int(W_raw)), None
        # Else we need to ceil / floor it
        H_0 = int(H_raw)
        H_1 = H_0
        if H_0 < H_raw:
            H_1 += 1
        W_0 = int(W_raw)
        W_1 = W_0
        if W_0 < W_raw:
            W_1 += 1
        # Return possible shapes
        is_h_ndiff = H_0 != H_1
        is_w_ndiff = W_0 != W_1
        ret = []
        ret.append((W_0, H_0))
        if is_h_ndiff:
            ret.append((W_0, H_1))
        if is_w_ndiff:
            ret.append((W_1, H_0))

        if is_h_ndiff and is_w_ndiff:
            ret.append((W_1, H_1))
        return ret

    def get_batch_shape(self, batch_idx: int) -> Tuple[int, int]:
        """Get the shape of the batch for the given batch index.

        Takes into account that B = H * W, where H, W are the image dimensions which are subsampled by the subsample factor,
        can be different as they may not be divisible by the subsample factor.

        Parameters
        ----------
        batch_idx : int
            Batch index to get the shape for B.

        Returnss
        -------
        Tuple[int, int]
            H, W of the batch for the given batch index.
        """
        resolution = self._resolution
        W_M, H_M = resolution.detach() // self._subsample

        is_h_out = (H_M * self._subsample[1] +
                    self._subsample_offsets[batch_idx, 1]) >= resolution[1]
        is_w_out = (W_M * self._subsample[0] +
                    self._subsample_offsets[batch_idx, 0]) >= resolution[0]

        W_raw, H_raw = resolution / self._subsample
        H = int(H_raw) if is_h_out else int(H_raw) + 1
        W = int(W_raw) if is_w_out else int(W_raw) + 1
        return H, W

    def get_proto_image_tensor(self, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the prototype image tensor for the given timestamps.

        This information can be used to query a image generation multiple times according to the subsample factor to form the full image.

        Parameters
        ----------
        t : Optional[torch.Tensor], optional
            Timestamps to evaluate for, by default None

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            1. Image tensor The image tensor with the shape (T, C, H, W)
            2. Subsample The subsample factor for the image tensor
            3. Subsample offsets The subsample offsets for the image tensor
        """
        if t is None:
            t = self._t
        W, H = self._resolution
        img = torch.zeros(len(t), 3, H, W, dtype=self.config.dtype)
        return img, self._subsample, self._subsample_offsets

    def sample_grid(
            self,
            batch_idx: int,
            flatten: bool = True,
            t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample grid for the given batch index.

        Parameters
        ----------

        batch_idx : int
            Index of the batch to sample rays for.
            Each batch contains a spatially subsampled set of rays e.g. 1/num_batches * image_dimension (H, W) of the image.

        flatten : bool, optional
            Whether to flatten the batch dimensions, by default True
            If false, the batch dimensions are kept. (H, W, 2) instead of (H * W, 2)

        t : Optional[torch.Tensor], optional
            Timestamps to get the object position for, by default None
            If none, the timestamps for the camera are used. Should be of shape (T,) in range [0, 1]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing:
            1. Sample grid
            The UV / Grid coordinates of the rays. Shape: (B, 2)

            2. Times
            The sample times at the time dimension. Shape: (T,)

            B is H * W, T is the number of time samples.
        """
        # UV coordinates
        offset = self._subsample_offsets[batch_idx]
        uv = sample_uv_grid(self.resolution,
                            self._subsample,
                            offset,
                            align_corners=self.align_corners,
                            uv_max=self.uv_max,
                            inter_pixel_noise_fnc=self._inter_pixel_noise_fnc,
                            dtype=self.config.dtype
                            )
        if t is None:
            t = self._t

        if flatten:
            uv, _ = flatten_batch_dims(uv, -2)
        return uv, t
