import math
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.learned_offset_scene_node_3d import LearnedOffsetSceneNode3D
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global

import torch
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix
from nag.config.encoding_config import EncodingConfig
from nag.config.network_config import NetworkConfig
from nag.transforms.transforms_timed_3d import interpolate_vector, linear_interpolate_vector
from tools.viz.matplotlib import plot_mask, plot_as_image
from tools.util.numpy import numpyify
from nag.utils import utils
try:
    import tinycudann as tcnn
except (ModuleNotFoundError, OSError) as err:
    from tools.logger.logging import logger
    from tools.util.mock_import import MockImport
    if not TYPE_CHECKING:
        logger.warning(f"Could not import tinycudann: {err}")
        tcnn = MockImport(mocked_property="tcnn")

import pytorch_lightning as pl
from tools.transforms.geometric.transforms3d import rotmat_to_unitquat, unitquat_to_rotmat
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates, plane_coordinates_to_local
from tools.util.torch import shadow_ones, shadow_zeros, shadow_identity_2d
from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D, default_control_point_times
from tools.util.format import raise_on_none
from tools.transforms.mean_std import MeanStd
from tools.util.typing import DEFAULT
from tools.transforms.fittable_transform import FittableTransform


class TimeDependentBackgroundImagePlaneSceneNode3D(
    BackgroundImagePlaneSceneNode3D
):

    encoding_time_dependence: tcnn.Encoding
    """Encoding for the time dependence."""

    network_time_dependence: tcnn.Network
    """Network for the time dependence."""

    time_dependence_weight: torch.Tensor
    """Weight for the view dependence."""

    def __init__(
        self,
            num_flow_control_points: int,
            num_time_dependent_control_points: Optional[int] = None,
            time_dependent_control_point_ratio: float = 0.1,
            encoding_time_dependence_config: Union[EncodingConfig,
                                                   str] = "small",
            network_time_dependence_config: Union[NetworkConfig,
                                                  str] = "small",
            time_dependence_weight: float = 0.1,
            time_dependence_rescaling: bool = True,
            time_dependence_normalization: Optional[FittableTransform] = None,
            proxy_init: bool = False,
            dtype: torch.dtype = torch.float32,
            network_dtype: torch.dtype = torch.float16,
            **kwargs
    ):
        super().__init__(
            num_flow_control_points=num_flow_control_points,
            network_dtype=network_dtype,
            dtype=dtype,
            proxy_init=proxy_init,
            **kwargs)

        time_dependence_input_dims = 2

        if num_time_dependent_control_points is None:
            times = kwargs.get("_times", None)
            if times is None:
                raise ValueError(
                    "num_time_dependent_control_points must be provided if _times is not.")
            num_time_dependent_control_points = int(
                round(len(times) * time_dependent_control_point_ratio))

        # RGB Spline for time
        time_dependence_output_dims = 3 * \
            (num_time_dependent_control_points + 2)
        self.num_time_dependent_control_points = num_time_dependent_control_points + 2

        self.register_buffer("time_dependent_timestamps", default_control_point_times(
            num_time_dependent_control_points, dtype=dtype))

        self.encoding_time_dependence = tcnn.Encoding(
            n_input_dims=time_dependence_input_dims,
            encoding_config=EncodingConfig.parse(
                encoding_time_dependence_config).to_dict(),
            dtype=network_dtype
        )

        self.network_time_dependence = tcnn.Network(
            n_input_dims=self.encoding_time_dependence.n_output_dims,
            n_output_dims=time_dependence_output_dims,
            network_config=NetworkConfig.parse(network_time_dependence_config).to_dict())

        self.register_buffer("time_dependence_weight", tensorify(
            time_dependence_weight, dtype=dtype, device=self._translation.device))

        self.time_dependence_rescaling = time_dependence_rescaling
        if self.time_dependence_rescaling:
            if time_dependence_normalization is None:
                time_dependence_normalization = MeanStd(
                    dim=0, mean=0, std=DEFAULT)
            self.time_dependence_normalization = time_dependence_normalization
            if not proxy_init:
                self.estimate_time_dependence_scaling(
                    time_dependence_normalization)
        else:
            self.time_dependence_normalization = None

    def after_checkpoint_loaded(self, **kwargs):
        super().after_checkpoint_loaded(**kwargs)
        if self.time_dependence_rescaling and self.time_dependence_normalization is not None:
            self.time_dependence_normalization.fitted = True

    def estimate_time_dependence_scaling(self, normalization: FittableTransform):
        H, W = 1000, 1000
        with torch.no_grad():
            device = torch.device("cuda")
            old_device = self._translation.device
            if device != old_device:
                self.to(device)
            x = torch.linspace(0, 1, W, device=device,
                               dtype=self._translation.dtype)
            y = torch.linspace(0, 1, H, device=device,
                               dtype=self._translation.dtype)
            grid = (torch.stack(torch.meshgrid(
                x, y, indexing="xy"), dim=-1)) - 0.5
            field = self._query_time_dependence(
                grid.reshape(H * W, 2).unsqueeze(1),
                torch.tensor(0, device=device, dtype=self._translation.dtype))  # (H * W, 1, TT, 3)
            _ = normalization.fit_transform(field.reshape(
                H * W * 1, self.num_time_dependent_control_points, 3))
            if device != old_device:
                self.to(old_device)

    def _query_time_dependence(self,
                               uv: torch.Tensor,
                               sin_epoch: torch.Tensor,
                               **kwargs) -> torch.Tensor:
        """Query the time dependence values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            UV coordinates of the points. Shape: (B, T, 2)
            In the range [-0.5, 0.5]


        sin_epoch : torch.Tensor
            Sine of the epoch. Shape: ()
            For coarse to fine training.

        Returns
        -------
        torch.Tensor
            Time dependence values for the given uv coordinates. Shape: (B, T, TT, 3)
        """
        input_coords = uv + 0.5  # (B, T, 2)
        input_coords = input_coords + 0.5
        B, T, _ = input_coords.shape
        TT = self.num_time_dependent_control_points
        input_coords = input_coords.reshape(B * T, 2)

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            time_dependence = self.network_time_dependence(utils.mask(
                self.encoding_time_dependence(input_coords), sin_epoch))  # (B, 2)

        time_dependence = time_dependence.to(
            dtype=self.dtype).reshape(B * T, TT, 3)

        if self.time_dependence_rescaling and self.time_dependence_normalization.fitted:
            time_dependence = self.time_dependence_normalization(
                time_dependence)

        time_dependence = time_dependence.reshape(B, T, TT, 3)

        return time_dependence

    def get_time_dependence(self,
                            uv: torch.Tensor,
                            t: torch.Tensor,
                            sin_epoch: torch.Tensor,
                            context: Optional[Dict[str, Any]] = None,
                            is_inside: Optional[torch.Tensor] = None,
                            **kwargs) -> torch.Tensor:
        """
        Get the view dependence values for the given uv coordinates and incline angles.

        Note: Time is not considered in the view dependence.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point and resp. time
            Shape: (B, T, 2) x, y should be in range [-0.5, 0.5]

        t : torch.Tensor
            The times of the points. Shape: (T, )

        Returns
        -------
        torch.Tensor
            The time dependence values for the given uv coordinates and incline angles. Represents the color (RGB), for the resp. position, angle and time.
            Shape: (B, T, 3)
        """
        B, T, _ = uv.shape
        TT = self.num_time_dependent_control_points
        BT = B * T

        time_query = t.unsqueeze(1).expand(B, T, 1)  # (B, T, 1)

        if is_inside is not None:
            uv = uv[is_inside].unsqueeze(1)
            time_query = time_query[is_inside]  # (BT, 1)
            BT = uv.shape[0]

        if uv.numel() != 0:
            value_out = self._query_time_dependence(
                uv, sin_epoch).reshape(BT, TT, 3)  # (BT, TT, 3)

            t_stamps = self.time_dependent_timestamps.unsqueeze(
                0).expand(BT, TT)  # (BT, TT)

            time_dependence = interpolate_vector(value_out,
                                                 t_stamps,
                                                 time_query,
                                                 equidistant_times=True,
                                                 method="cubic")  # (BT, 1, 3)

        else:
            time_dependence = torch.zeros(
                BT, 1, 3, dtype=uv.dtype, device=uv.device)
        if is_inside is None:
            return time_dependence.reshape(B, T, 3)
        else:
            ret = torch.zeros(B, T, 3, dtype=time_dependence.dtype,
                              device=time_dependence.device)
            ret[is_inside] = time_dependence[:, 0]
            return ret

    # region Forward

    def compute_rgb_alpha(self,
                          uv: torch.Tensor,
                          t: torch.Tensor,
                          sin_epoch: torch.Tensor,
                          right_idx_flow: Optional[torch.Tensor] = None,
                          rel_frac_flow: Optional[torch.Tensor] = None,
                          ray_origins: Optional[torch.Tensor] = None,
                          ray_directions: Optional[torch.Tensor] = None,
                          context: Optional[Dict[str, Any]] = None,
                          **kwargs
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the RGB and alpha values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            UV coordinates of the points within the image plane. Shape: (B, T, 2) (x, y) in range [-0.5, 0.5]
        t : torch.Tensor
            The times of the points. Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch. Progress marker. Tends towards 1 at the end of training.
        right_idx_flow : Optional[torch.Tensor], optional
            Index for position interpolation, by default None
        rel_frac_flow : Optional[torch.Tensor], optional
            Step for position interpolation, by default None
        ray_origins : Optional[torch.Tensor], optional
            The ray origins of the rays. In local coordinates of the current node. Shape: (B, T, 3)
        ray_directions : Optional[torch.Tensor], optional
            The ray directions of the rays. In local coordinates of the current node. Shape: (B, T, 3)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            RGB values and alpha values for the given uv coordinates. Shapes: (B, T, 3), (B, T, 1)
            Whereby both are in range [0, 1].
        """
        B, T, _ = uv.shape
        # In local coordinates the plane center is at (0, 0, 0)
        if self.flow_weight != 0:
            if self.deprecated_flow:
                flow = self._compute_flow(uv, t, sin_epoch,
                                          right_idx_flow=right_idx_flow,
                                          rel_frac_flow=rel_frac_flow
                                          )
            else:
                flow = self.get_flow(uv, t, sin_epoch,
                                     right_idx_flow=right_idx_flow,
                                     rel_frac_flow=rel_frac_flow
                                     )
        else:
            flow = torch.zeros_like(uv)

        query_points = uv + flow  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        # And the flow as adjusted for the time already
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)

        time_dependence = self.get_time_dependence(
            query_points.reshape(B, T, 2),
            t=t,
            sin_epoch=sin_epoch,
            context=context
        ).reshape(-1, 3)  # B, 3
        time_rgb = time_dependence[:, :3]

        # Get the RGB values
        network_rgb = self.get_rgb(query_points, sin_epoch)
        rgb = self.get_initial_rgb(query_points) + \
            self.rgb_weight * network_rgb + self.time_dependence_weight * time_rgb

        rgb = rgb.clamp(0, 1)
        rgb = rgb.reshape(B, T, 3)

        # Get the alpha values
        alpha = torch.ones(B, T, 1, dtype=self.dtype,
                           device=uv.device)

        if context is not None:
            idx = self.get_index()
            if context.get("store_object_flow", False):
                if "object_flow" not in context:
                    context["object_flow"] = dict()
                context["object_flow"][idx] = flow

        return rgb, alpha

    def forward_modality(
        self,
        uv_plane: torch.Tensor,
        t: torch.Tensor,
        sin_epoch: torch.Tensor,
        right_idx_flow: Optional[torch.Tensor] = None,
        rel_frac_flow: Optional[torch.Tensor] = None,
        next_sin_epoch: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        max_batch_idx: Optional[int] = None,
        query_alpha: bool = True,
        query_color: bool = True,
        ray_origins: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the plane.

        Parameters
        ----------
        uv_plane : torch.Tensor
            The intersection points / query points of the rays with the plane.
            In plane space. Plane space is in normal range [0, 1].
            Shape: (B, T, 2)

        t : torch.Tensor
            The time of the intersection.
            Shape: (T, )

        sin_epoch : torch.Tensor
            The sine of the epoch.

        right_idx_flow : Optional[torch.Tensor], optional
            Index for flow controlpoint interpolation for times t, by default None

        rel_frac_flow : Optional[torch.Tensor], optional
            Step for position interpolation, by default None

        next_sin_epoch : Optional[torch.Tensor], optional
            The sine of the next epoch, by default None

        batch_idx : Optional[int], optional
            The batch index, by default None

        max_batch_idx : Optional[int], optional
            The maximum batch index, by default None

        query_alpha : bool, optional
            If True, the alpha values are queried, by default True

        query_color : bool, optional
            If True, the color values are queried, by default True

        ray_origins : Optional[torch.Tensor], optional
            The ray origins of the rays. In local coordinates of the current node. Shape: (B, T, 3)

        ray_directions : Optional[torch.Tensor], optional
            The ray directions of the rays. In local coordinates of the current node. Shape: (B, T, 3)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            1. The RGB values of the plane. Shape: (B, T, 3)
            2. The alpha values of the plane. Shape: (B, T, 1)
            3. The flow values of the plane. Shape: (B, T, 2)
        """

        uv_network = uv_plane - 0.5

        B, T, _ = uv_network.shape
        # In local coordinates the plane center is at (0, 0, 0)
        if self.flow_weight != 0:
            if self.deprecated_flow:
                flow = self._compute_flow(uv_network, t, sin_epoch,
                                          right_idx_flow=right_idx_flow,
                                          rel_frac_flow=rel_frac_flow
                                          )
            else:
                flow = self.get_flow(uv_network, t, sin_epoch,
                                     right_idx_flow=right_idx_flow,
                                     rel_frac_flow=rel_frac_flow
                                     )
        else:
            flow = torch.zeros_like(uv_network)

        query_points = uv_network + flow  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)
        # query_points = shadow_zeros(query_points)

        if query_color:
            if ray_directions is None:
                raise ValueError(
                    "Ray directions must be provided for view dependence.")

            time_dependence = self.get_time_dependence(query_points.reshape(B, T, 2),
                                                       t=t,
                                                       sin_epoch=sin_epoch).reshape(-1, 3)

            # Get the RGB values
            network_rgb = self.get_rgb(query_points, sin_epoch)
            rgb = self.get_initial_rgb(query_points) + \
                self.rgb_weight * network_rgb + self.time_dependence_weight * time_dependence
            rgb = rgb.clamp(0, 1)
            # Reshape to original shape
            rgb = rgb.reshape(B, T, 3)
        else:
            rgb = torch.zeros(B, T, 3, dtype=self.dtype,
                              device=uv_plane.device)

        alpha = torch.zeros(B, T, 1, dtype=self.dtype,
                            device=uv_plane.device)

        return rgb, alpha, flow

    # endregion
