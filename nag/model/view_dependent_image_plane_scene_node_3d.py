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
from nag.model.learned_color_alpha_location_image_plane_scene_node_3d import LearnedColorAlphaLocationImagePlaneSceneNode3D
from tools.util.format import raise_on_none
from tools.transforms.mean_std import MeanStd
from tools.util.typing import DEFAULT
from tools.transforms.fittable_transform import FittableTransform
from nag.model.discrete_plane_scene_node_3d import compute_incline_angle


class ViewDependentImagePlaneSceneNode3D(
    LearnedColorAlphaLocationImagePlaneSceneNode3D
):

    encoding_view_dependence: tcnn.Encoding
    """Encoding for the view dependence."""

    network_view_dependence: tcnn.Network
    """Network for the view dependence."""

    view_dependence_weight: torch.Tensor
    """Weight for the view dependence."""

    def __init__(
        self,
            num_rigid_control_points: int,
            num_flow_control_points: int,
            encoding_view_dependence_config: Union[EncodingConfig,
                                                   str] = "small",
            network_view_dependence_config: Union[NetworkConfig,
                                                  str] = "small",
            view_dependence_weight: float = 0.1,
            view_dependence_rescaling: bool = True,
            view_dependence_input_dims: int = 4,
            view_dependence_output_dims: int = 4,
            view_dependence_normalization: Optional[FittableTransform] = None,
            proxy_init: bool = False,
            view_dependence_normalization_init: bool = True,
            dtype: torch.dtype = torch.float32,
            network_dtype: torch.dtype = torch.float16,
            **kwargs
    ):
        super().__init__(
            num_rigid_control_points=num_rigid_control_points,
            num_flow_control_points=num_flow_control_points,
            network_dtype=network_dtype,
            dtype=dtype,
            proxy_init=proxy_init,
            **kwargs)
        self.ray_dependent = True

        self.encoding_view_dependence = tcnn.Encoding(
            n_input_dims=view_dependence_input_dims,
            encoding_config=EncodingConfig.parse(
                encoding_view_dependence_config).to_dict(),
            dtype=network_dtype
        )

        self.network_view_dependence = tcnn.Network(
            n_input_dims=self.encoding_view_dependence.n_output_dims,
            n_output_dims=view_dependence_output_dims,
            network_config=NetworkConfig.parse(network_view_dependence_config).to_dict())

        self.register_buffer("view_dependence_weight", tensorify(
            view_dependence_weight, dtype=dtype, device=self._translation.device))

        self.view_dependence_rescaling = view_dependence_rescaling
        if self.view_dependence_rescaling:
            if view_dependence_normalization is None:
                view_dependence_normalization = MeanStd(
                    dim=0, mean=0, std=DEFAULT)
            self.view_dependence_normalization = view_dependence_normalization
            if not proxy_init and view_dependence_normalization_init:
                self.estimate_view_dependence_scaling(
                    view_dependence_normalization)
        else:
            self.view_dependence_normalization = None

    def after_checkpoint_loaded(self, **kwargs):
        super().after_checkpoint_loaded(**kwargs)
        if self.view_dependence_rescaling and self.view_dependence_normalization is not None:
            self.view_dependence_normalization.fitted = True

    def estimate_view_dependence_scaling(self, normalization: FittableTransform):
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

            view_field = self.get_view_dependence(uv_vec[:, None, :],
                                                  angle_vec[:, None, :],
                                                  t=torch.tensor(
                                                      [0.], device=device, dtype=self._translation.dtype),
                                                  sin_epoch=torch.tensor(0., device=device, dtype=self._translation.dtype)).reshape(V * H * W, 4)
            view_dep_norm = normalization.fit_transform(view_field)
            if device != old_device:
                self.to(old_device)

    def get_view_dependence(self,
                            uv: torch.Tensor,
                            angle: torch.Tensor,
                            t: torch.Tensor,
                            sin_epoch: torch.Tensor,
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

        angle : torch.Tensor
            The incline angles of the uv (intersection) points at the resp. time.
            Should be the angle w.r.t the normal (z+) of the plane.
            Shape: (B, T, 2) angles should be in range [-pi, pi]

        t : torch.Tensor
            The times of the points. Shape: (T, )

        Returns
        -------
        torch.Tensor
            The view dependence values for the given uv coordinates and incline angles. Represents the color (RGB) and A offset, for the resp. position, angle and time.
            Shape: (B, T, 4)
        """
        coords = uv * 2 * \
            torch.pi  # (B, T, 2) Convert the uv coordinates in the same value range as the angles
        input_coords = torch.cat([coords, angle], dim=-1)  # (B, T, 4)

        B, T, _ = input_coords.shape
        input_coords = input_coords.reshape(B * T, 4)

        if not self.legacy_model_inputs:
            # Convert tensor to 0-1 range
            input_coords = (input_coords + torch.pi) / (2 * torch.pi)

        if is_inside is not None:
            input_coords = input_coords[is_inside.reshape(B * T)]

        if input_coords.numel() == 0:
            return torch.zeros((B, T, 4), dtype=self.dtype, device=uv.device)

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            view_dependence = self.network_view_dependence(utils.mask(
                self.encoding_view_dependence(input_coords), sin_epoch))  # (B, 2)

        view_dependence = view_dependence.to(dtype=self.dtype)

        if self.view_dependence_rescaling and self.view_dependence_normalization.fitted:
            view_dependence = self.view_dependence_normalization(
                view_dependence)

        if is_inside is not None:
            complete_view_dependence = torch.zeros(
                (B, T, 4), dtype=self.dtype, device=view_dependence.device)
            complete_view_dependence[is_inside] = view_dependence
            view_dependence = complete_view_dependence

        view_dependence = view_dependence.reshape(B, T, 4)

        return view_dependence

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
                          is_inside: Optional[torch.Tensor] = None,
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
                                     rel_frac_flow=rel_frac_flow,
                                     is_inside=is_inside
                                     )
            if self.independent_rgba_flow:
                flow_alpha = self.get_flow_alpha(uv, t, sin_epoch,
                                                 right_idx_flow=right_idx_flow,
                                                 rel_frac_flow=rel_frac_flow,
                                                 is_inside=is_inside
                                                 )
        else:
            flow = torch.zeros_like(uv)
            if self.independent_rgba_flow:
                flow_alpha = torch.zeros_like(uv)

        query_points = uv + flow  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        # And the flow as adjusted for the time already
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)

        if self.independent_rgba_flow:
            query_points_alpha = uv + flow_alpha  # (B, T, 2)
            query_points_alpha = query_points_alpha.reshape(-1, 2)  # (B*T, 2)
        else:
            query_points_alpha = query_points

        if self.alpha_rigid_model:
            # Apply the rigid shift to the alpha plane
            alpha_shift = self.get_rigid_alpha(uv, t)
            query_points_alpha = query_points_alpha + \
                alpha_shift.reshape(-1, 2)

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

        view_dependence = self.get_view_dependence(query_points.reshape(B, T, 2),
                                                   angle=rot_vec[..., :2],
                                                   t=t,
                                                   sin_epoch=sin_epoch,
                                                   context=context,
                                                   is_inside=is_inside
                                                   ).reshape(-1, 4)  # B, 4
        view_rgb = view_dependence[:, :3]
        view_alpha = view_dependence[:, 3:4]

        if is_inside is not None:
            query_points = query_points[is_inside.reshape(B * T)]
            query_points_alpha = query_points_alpha[is_inside.reshape(B * T)]
            view_rgb = view_rgb[is_inside.reshape(B * T)]
            view_alpha = view_alpha[is_inside.reshape(B * T)]

        if query_points.numel() != 0:
            # Get the RGB values
            network_rgb = self.get_rgb(query_points, sin_epoch)
            rgb = (
                self.get_initial_rgb(query_points) +
                self.rgb_weight * network_rgb + self.view_dependence_weight * view_rgb)

            rgb = rgb.clamp(0, 1)
            if self.render_texture_map:
                rgb = self.get_rendered_texture_map(query_points, rgb)

            # Get the alpha values
            network_alpha = self.get_alpha(query_points_alpha, sin_epoch)
            initial_alpha = self.get_initial_alpha(query_points_alpha)

            alpha = torch.sigmoid(
                (-torch.log(1/initial_alpha - 1)
                 + self.alpha_weight * network_alpha
                 + self.view_dependence_weight * view_alpha))
        else:
            rgb = torch.zeros((0, 3), dtype=self.dtype, device=uv.device)
            alpha = torch.zeros((0, 1), dtype=self.dtype, device=uv.device)

        if is_inside is not None:
            complete_rgb = torch.zeros(
                (B, T, 3), dtype=self.dtype, device=rgb.device)
            complete_alpha = torch.zeros(
                (B, T, 1), dtype=self.dtype, device=alpha.device)
            complete_rgb[is_inside] = rgb
            complete_alpha[is_inside] = alpha

            rgb = complete_rgb
            alpha = complete_alpha

        rgb = rgb.reshape(B, T, 3)
        alpha = alpha.reshape(B, T, 1)

        if context is not None:
            # Store the alpha and flow values in the context for regularization
            idx = self.get_index()
            if context.get("store_object_alpha", False):
                if "object_alpha" not in context:
                    context["object_alpha"] = dict()
                context["object_alpha"][idx] = alpha
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
        is_inside: Optional[torch.Tensor] = None,
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
                                     rel_frac_flow=rel_frac_flow,
                                     is_inside=is_inside
                                     )
            if self.independent_rgba_flow:
                flow_alpha = self.get_flow_alpha(uv_network, t, sin_epoch,
                                                 right_idx_flow=right_idx_flow,
                                                 rel_frac_flow=rel_frac_flow,
                                                 is_inside=is_inside
                                                 )
        else:
            flow = torch.zeros_like(uv_network)
            if self.independent_rgba_flow:
                flow_alpha = torch.zeros_like(uv_network)

        query_points = uv_network + flow  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)
        # query_points = shadow_zeros(query_points)

        if self.independent_rgba_flow:
            query_points_alpha = uv_network + flow_alpha
            query_points_alpha = query_points_alpha.reshape(-1, 2)  # (B*T, 2)
        else:
            query_points_alpha = query_points

        if self.alpha_rigid_model:
            # Apply the rigid shift to the alpha plane
            alpha_shift = self.get_rigid_alpha(uv_network, t)
            query_points_alpha = query_points_alpha + \
                alpha_shift.reshape(-1, 2)

        view_rgb = None
        view_alpha = None

        if query_color or query_alpha:
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

            view_dependence = self.get_view_dependence(query_points.reshape(B, T, 2),
                                                       angle=rot_vec[..., :2],
                                                       t=t,
                                                       sin_epoch=sin_epoch,
                                                       is_inside=is_inside
                                                       ).reshape(-1, 4)
            view_rgb = view_dependence[:, :3]
            view_alpha = view_dependence[:, 3:4]

            if is_inside is not None:
                query_points = query_points[is_inside.reshape(B * T)]
                view_rgb = view_rgb[is_inside.reshape(B * T)]
                view_alpha = view_alpha[is_inside.reshape(B * T)]

        else:
            view_rgb = torch.zeros(B * T, 3, dtype=self.dtype,
                                   device=uv_plane.device)
            view_alpha = torch.zeros(B * T, 1, dtype=self.dtype,
                                     device=uv_plane.device)

        if query_color:
            # Get the RGB values
            if query_points.numel() != 0:
                network_rgb = self.get_rgb(query_points, sin_epoch)
                rgb = self.get_initial_rgb(query_points) + \
                    self.rgb_weight * network_rgb + self.view_dependence_weight * view_rgb
                rgb = rgb.clamp(0, 1)
            else:
                rgb = torch.zeros(0, 3, dtype=self.dtype,
                                  device=uv_plane.device)
            # Reshape to original shape

            if is_inside is not None:
                complete_rgb = torch.zeros(
                    (B, T, 3), dtype=self.dtype, device=rgb.device)
                complete_rgb[is_inside] = rgb
                rgb = complete_rgb

            rgb = rgb.reshape(B, T, 3)
        else:
            rgb = torch.zeros(B, T, 3, dtype=self.dtype,
                              device=uv_plane.device)

        if query_alpha:
            # Get the alpha values
            if query_points.numel() != 0:
                network_alpha = self.get_alpha(query_points_alpha, sin_epoch)
                initial_alpha = self.get_initial_alpha(query_points_alpha)

                alpha = torch.sigmoid(
                    (-torch.log(1/initial_alpha - 1) + self.alpha_weight * network_alpha + self.view_dependence_weight * view_alpha))
            else:
                alpha = torch.zeros(0, 1, dtype=self.dtype,
                                    device=uv_plane.device)

            if is_inside is not None:
                complete_alpha = torch.zeros(
                    (B, T, 1), dtype=self.dtype, device=alpha.device)
                complete_alpha[is_inside] = alpha
                alpha = complete_alpha

            alpha = alpha.reshape(B, T, 1)
        else:
            alpha = torch.zeros(B, T, 1, dtype=self.dtype,
                                device=uv_plane.device)

        return rgb, alpha, flow

    # endregion
