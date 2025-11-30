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
from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D
from nag.model.view_dependent_background_image_plane_scene_node_3d import ViewDependentBackgroundImagePlaneSceneNode3D
from tools.util.format import raise_on_none
from tools.transforms.mean_std import MeanStd
from tools.util.typing import DEFAULT
from tools.transforms.fittable_transform import FittableTransform
from nag.model.discrete_plane_scene_node_3d import compute_incline_angle


class ViewDependentBackgroundLightfieldSceneNode3D(
    ViewDependentBackgroundImagePlaneSceneNode3D
):

    def __init__(
        self,
            num_flow_control_points: int,
            proxy_init: bool = False,
            dtype: torch.dtype = torch.float32,
            network_dtype: torch.dtype = torch.float16,
            **kwargs
    ):
        super().__init__(
            num_flow_control_points=num_flow_control_points,
            flow_input_dims=4,
            flow_normalization_init=False,
            flow_has_control_points=False,
            network_dtype=network_dtype,
            dtype=dtype,
            proxy_init=proxy_init,
            **kwargs)
        if not proxy_init:
            self.estimate_flow_scaling(
                self.flow_normalization)

    def estimate_flow_scaling(self, normalization: FittableTransform):
        V_N = 6
        V = V_N ** 2
        # Considering 1e6 points for the estimation, and 6 different view directions per axis
        H, W = int(math.floor(math.sqrt(1e6 / V))
                   ), int(math.floor(math.sqrt(1e6 / V)))
        with torch.no_grad():
            device = torch.device("cuda")
            old_device = self._translation.device
            if device != old_device:
                self.to(device)
            x = torch.linspace(0, 1, W, device=device,
                               dtype=self._translation.dtype) - 0.5
            y = torch.linspace(0, 1, H, device=device,
                               dtype=self._translation.dtype) - 0.5
            a1 = torch.arange(-torch.pi, torch.pi, 2 * torch.pi /
                              V_N, device=device, dtype=self._translation.dtype)
            a2 = torch.arange(-torch.pi, torch.pi, 2 * torch.pi /
                              V_N, device=device, dtype=self._translation.dtype)

            grid = (torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1))
            uv_vec = grid.reshape(
                H * W, 2).unsqueeze(0).repeat(V, 1, 1).reshape(V * H * W, 2)
            angle_vec = torch.stack(torch.meshgrid(a1, a2, indexing="xy"), dim=-1).reshape(
                V, 2).unsqueeze(1).repeat(1, H * W, 1).reshape(V * H * W, 2)

            flow_field = self.get_flow(uv_vec[:, None, :],
                                       angle_vec[:, None, :],
                                       t=torch.tensor(
                [0.], device=device, dtype=self._translation.dtype),
                sin_epoch=torch.tensor(0., device=device, dtype=self._translation.dtype)).reshape(V * H * W, 2)
            _ = normalization.fit_transform(flow_field)
            if device != old_device:
                self.to(old_device)

    def get_flow(self,
                 uv: torch.Tensor,
                 angle: torch.Tensor,
                 t: torch.Tensor,
                 sin_epoch: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Computes the flow for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of a point within the image plane.
            Shape: (B, T, 2) (x, y) in range [-0.5, 0.5]
        angle : torch.Tensor
            The Incline angles of the points. Shape: (B, T, 2)
            The angles are in radians and should be in the range [-pi, pi].
        t : torch.Tensor
            The times of the points. Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch, progress marker.

        Returns
        -------
        torch.Tensor
            The flow offset for the given uv coordinates. Shape: (B, T, 2)
        """
        coords = uv * 2 * \
            torch.pi  # (B, T, 2) Convert the uv coordinates in the same value range as the angles
        input_coords = torch.cat([coords, angle], dim=-1)  # (B, T, 4)

        B, T, _ = input_coords.shape
        input_coords = input_coords.reshape(B * T, 4)

        # Convert tensor to 0-1 range
        input_coords = (input_coords + torch.pi) / (2 * torch.pi)

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            flow = self.network_flow(utils.mask(
                self.encoding_flow(input_coords), sin_epoch))  # (B, 2)

        flow = flow.to(dtype=self.dtype)

        if self.flow_rescaling and self.flow_normalization.fitted:
            flow = self.flow_normalization(flow)

        flow = flow.reshape(B, T, 2)
        return flow

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
        if ray_directions is None:
            raise ValueError(
                "Ray directions must be provided for view dependence.")

        # Get view dependence
        rot_vec = compute_incline_angle(ray_directions, torch.tensor(
            [0, 0, 1], device=ray_directions.device, dtype=ray_directions.dtype))
        # Raise if any z is larger than

        zero_ang = torch.isclose(
            torch.nan_to_num(rot_vec[..., 2]), torch.zeros_like(rot_vec[..., 2]), atol=1e-5)
        assert zero_ang.all(
        ), f"Incline angle is not correct. z should be 0. But is: {rot_vec[~zero_ang][..., 2]}"

        antiparallel = torch.isclose(rot_vec[..., :2], torch.tensor(
            torch.pi, dtype=rot_vec.dtype, device=rot_vec.device), atol=1e-6).any(dim=-1)
        # Angle is pi, so we have an antiparallel vector. Set to 0
        rot_vec[antiparallel] = torch.zeros(
            3, dtype=rot_vec.dtype, device=rot_vec.device)

        B, T, _ = uv.shape
        # In local coordinates the plane center is at (0, 0, 0)
        if self.flow_weight != 0:
            flow = self.get_flow(uv, t=t, sin_epoch=sin_epoch,
                                 angle=rot_vec[..., :2],
                                 right_idx_flow=right_idx_flow,
                                 rel_frac_flow=rel_frac_flow
                                 )
        else:
            flow = torch.zeros_like(uv)

        query_points = uv + flow  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        # And the flow as adjusted for the time already
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)

        view_dependence = self.get_view_dependence(query_points.reshape(B, T, 2),
                                                   angle=rot_vec[..., :2],
                                                   t=t,
                                                   sin_epoch=sin_epoch,
                                                   context=context
                                                   ).reshape(-1, 3)  # B, 3
        view_rgb = view_dependence[:, :3]

        # Get the RGB values
        network_rgb = self.get_rgb(query_points, sin_epoch)
        rgb = self.get_initial_rgb(query_points) + \
            self.rgb_weight * network_rgb + self.view_dependence_weight * view_rgb
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

        if ray_directions is None:
            raise ValueError(
                "Ray directions must be provided for view dependence.")

        # Get view dependence
        rot_vec = compute_incline_angle(ray_directions, torch.tensor(
            [0, 0, 1], device=ray_directions.device, dtype=ray_directions.dtype))
        # Raise if any z is larger than
        assert torch.allclose(rot_vec[..., 2], torch.zeros_like(
            rot_vec[..., 2]), atol=1e-5), "Incline angle is not correct. z should be 0."

        antiparallel = torch.isclose(rot_vec[..., :2], torch.tensor(
            torch.pi, dtype=rot_vec.dtype, device=rot_vec.device), atol=1e-6).any(dim=-1)
        # Angle is pi, so we have an antiparallel vector. Set to 0
        rot_vec[antiparallel] = torch.zeros(
            3, dtype=rot_vec.dtype, device=rot_vec.device)

        # In local coordinates the plane center is at (0, 0, 0)
        if self.flow_weight != 0:
            flow = self.get_flow(uv_network, t=t, sin_epoch=sin_epoch,
                                 angle=rot_vec[..., :2],
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
            view_dependence = self.get_view_dependence(query_points.reshape(B, T, 2),
                                                       angle=rot_vec[..., :2],
                                                       t=t,
                                                       sin_epoch=sin_epoch).reshape(-1, 3)

            # Get the RGB values
            network_rgb = self.get_rgb(query_points, sin_epoch)
            rgb = self.get_initial_rgb(query_points) + \
                self.rgb_weight * network_rgb + self.view_dependence_weight * view_dependence
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
