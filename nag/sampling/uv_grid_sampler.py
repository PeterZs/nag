import abc
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
from tools.mixin.torch_dtype_mixin import TorchDtypeMixin


class UVGridSampler(Dataset, TorchDtypeMixin):

    _t: torch.Tensor
    """Timestamps for the ray positions. Shape is (T,) where T is the number of timestamps."""

    _uv_max: torch.Tensor
    """Maximum UV coordinate for the grid. Shape is (2,) where 2 is the (x, y) coordinate.
    This is used to scale the UV coordinates to the image resolution, if imaging is done for a smaller resolution.
    """

    _resolution: Tuple[int, int]
    """Resolution of the image to sample for the grid. (W, H) (x, y)"""

    _flatten: bool = True
    """Whether to flatten the batch dimensions, by default True"""

    def __init__(self,
                 resolution: Tuple[int, int],
                 uv_max: Tuple[int, int],
                 t: torch.Tensor,
                 dtype: torch.dtype = torch.float32,
                 **kwargs
                 ) -> None:
        super().__init__(dtype=dtype, **kwargs)
        self._t = tensorify(raise_on_none(t))
        self._uv_max = tensorify(raise_on_none(uv_max))
        self._resolution = tensorify(raise_on_none(resolution))

    @classmethod
    def for_camera(cls,
                   camera: TimedCameraSceneNode3D,
                   resolution_factor: float = 1.) -> 'UVGridSampler':
        """Create a new UVGridSampler for the given camera."""
        resolution = camera._image_resolution.flip(
            -1).detach().cpu().numpy()
        uv_max = camera._image_resolution.flip(
            -1).detach().cpu().numpy()
        t = camera._times.detach().cpu().clone()
        resolution = (resolution_factor * resolution).astype(int)
        return cls(resolution=resolution, uv_max=uv_max, t=t, dtype=camera.dtype)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def t(self) -> torch.Tensor:
        return self._t

    @t.setter
    def t(self, t: torch.Tensor) -> None:
        self._t = t

    @property
    def uv_max(self) -> torch.Tensor:
        """UV max for the grid. Shape is (2,) where 2 is the (x, y) coordinate."""
        return self._uv_max

    @property
    def flatten(self) -> bool:
        return self._flatten

    @flatten.setter
    def flatten(self, flatten: bool) -> None:
        self._flatten = flatten

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_grid(idx, flatten=self._flatten, t=self._t)

    @abc.abstractmethod
    def sample_grid(
            self,
            batch_idx: int,
            flatten: bool = True,
            t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        """
        raise NotImplementedError()

    def plot_rays(self, batch_idx: Union[int, List[int]], camera: TimedCameraSceneNode3D) -> Figure:
        """Plot the rays for the given batch index."""
        # Plot camera
        if isinstance(batch_idx, int):
            batch_idx = [batch_idx]

        t = 0.
        t_idx = 0

        with torch.no_grad():
            ro, rd = list(), list()
            for idx in batch_idx:
                out = self.sample_grid(idx)
                uv = out[0]
                t = out[-1]
                # Get the rays
                ro_, rd_ = camera.get_global_rays(
                    uv, t, uv_includes_time=False)

                ro.append(ro_[..., t_idx, :])
                rd.append(rd_[..., t_idx, :])

        fig = camera.plot_scene(t=t)
        ax = fig.axes[0]

        cmap = plt.get_cmap('tab20') if len(
            batch_idx) > 10 else plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(batch_idx))]

        for i, (o, d) in enumerate(zip(ro, rd)):
            camera._plot_rays(o, d, ax=ax, color=colors[i])
        return fig
