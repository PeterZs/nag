import math
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.learned_offset_scene_node_3d import LearnedOffsetSceneNode3D
from nag.model.timed_box_scene_node_3d import TimedBoxSceneNode3D
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
from tools.transforms.min_max import MinMax
from tools.transforms.mean_std import MeanStd
from tools.transforms.fittable_transform import FittableTransform
from tools.util.typing import DEFAULT
import torch.nn.functional as F
from nag.model.texture_mappable_scene_node_3d import TextureMappableSceneNode3D


def minmax(v: torch.Tensor,
           v_min: torch.Tensor,
           v_max: torch.Tensor,
           new_min: torch.Tensor,
           new_max: torch.Tensor,
           ) -> torch.Tensor:
    return (v - v_min) / (v_max - v_min) * (new_max - new_min) + new_min


def default_control_point_times(
        num_control_points: int,
        dtype: torch.dtype) -> torch.Tensor:
    if num_control_points > 1:
        t_step = 1. / (num_control_points - 1)
        return torch.linspace(0 - t_step, 1 + t_step, num_control_points + 2, dtype=dtype)
    else:
        return torch.tensor([-1, 0, 1], dtype=dtype)


@torch.jit.script
def global_to_local_rays(
        global_position: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Parameters
    ----------
    global_position : torch.Tensor
        _description_
    ray_origins : torch.Tensor
        _description_
    ray_directions : torch.Tensor
        _description_

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        _description_
    """

    ray_global_target = ray_origins + ray_directions
    ray_local_target = global_to_local(
        global_position, ray_global_target, v_include_time=True)[..., :3]
    ray_local_origin = global_to_local(
        global_position, ray_origins, v_include_time=True)[..., :3]
    ray_local_directions = ray_local_target - ray_local_origin
    return ray_local_origin, ray_local_directions


class LearnedImagePlaneSceneNode3D(
    LearnedOffsetSceneNode3D,
    TextureMappableSceneNode3D
):
    """A scene node representing a image plane in 3D space with timed discrete position and orientation and learnable offsets.
    And an encoding and
    """

    encoding_image: tcnn.Encoding
    """The encoding for the image plane"""

    encoding_alpha: tcnn.Encoding
    """The encoding for the alpha matting of the plane."""

    encoding_flow: tcnn.Encoding
    """The encoding for the flow of the plane."""

    num_flow_control_points: int
    """The number of control points for the flow of the plane. The optical flow is discretized into this number of control points."""

    flow_timestamps: torch.Tensor
    """The timestamps for the flow control points. Shape: (num_flow_control_points, ) in range [0, 1]."""

    flow_weight: torch.Tensor
    """The weight factor for the flow offset. Shape (1,) / float."""

    flow_reference_time: torch.Tensor
    """The reference time for the flow interpolation. If None, the time 0 is used."""

    rgb_weight: torch.Tensor
    """The weight factor for the RGB offset. Shape (1,) / float."""

    alpha_weight: torch.Tensor
    """The weight factor for the alpha offset. Shape (1,) / float."""

    network_image: tcnn.Network
    """The tcnn network for predicting the image color plane."""

    network_alpha: tcnn.Network
    """The tcnn network for predicting the alpha of the plane."""

    network_flow: tcnn.Network
    """The tcnn network for predicting the flow on the plane."""

    network_dtype: torch.dtype
    """The dtype for the network."""

    indenpendent_rgba_flow: bool
    """If True, the flow is computed independently for RGB and alpha. If False, the flow is computed jointly for RGB and alpha."""

    network_flow_alpha: Optional[tcnn.Network]
    """The tcnn network for predicting the flow on the alpha plane. Only used if `indenpendent_rgba_flow` is True."""

    encoding_flow_alpha: Optional[tcnn.Encoding]
    """The encoding for the flow on the alpha plane. Only used if `indenpendent_rgba_flow` is True."""

    def __init__(
        self,
            num_rigid_control_points: int,
            num_flow_control_points: int,
            encoding_image_config: Union[EncodingConfig, str] = "small",
            encoding_alpha_config: Union[EncodingConfig, str] = "small",
            encoding_flow_config: Union[EncodingConfig, str] = "small",
            network_image_config: Union[NetworkConfig, str] = "small",
            network_alpha_config: Union[NetworkConfig, str] = "small",
            network_flow_config: Union[NetworkConfig, str] = "small",
            translation_offset_weight: float = 0.03,
            rotation_offset_weight: float = 0.03,
            rgb_rescaling: bool = True,
            rgb_normalization: Optional[FittableTransform] = None,
            alpha_rescaling: bool = True,
            alpha_normalization: Optional[FittableTransform] = None,
            flow_weight: float = 0.01,
            flow_reference_time: Optional[VEC_TYPE] = None,
            flow_rescaling: bool = True,
            flow_normalization: Optional[FittableTransform] = None,
            coarse_to_fine_color: bool = True,
            coarse_to_fine_alpha: bool = True,
            plane_scale: Optional[VEC_TYPE] = None,
            plane_scale_offset: Optional[VEC_TYPE] = None,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            initial_rgb: Optional[torch.Tensor] = None,
            initial_rgb_variance: Optional[torch.Tensor] = None,
            mean_rgb_variance: Optional[torch.Tensor] = None,
            initial_alpha: Optional[torch.Tensor] = None,
            rgb_weight: Optional[torch.Tensor] = None,
            alpha_weight: Optional[torch.Tensor] = None,
            box: Optional[TimedBoxSceneNode3D] = None,
            position: Optional[torch.Tensor] = None,
            times: Optional[VEC_TYPE] = None,
            name: Optional[str] = None,
            children: Optional[Iterable['AbstractSceneNode']] = None,
            decoding: bool = False,
            dtype: torch.dtype = torch.float32,
            network_dtype: torch.dtype = torch.float16,
            index: Optional[int] = None,
            # Proxy init is used if the class should be initialized, but values may be dummies as weights are loaded afterwards
            proxy_init: bool = False,
            deprecated_flow: bool = True,
            independent_rgba_flow: bool = False,
            alpha_rigid_model: bool = False,
            **kwargs
    ):
        super().__init__(
            num_control_points=num_rigid_control_points,
            translation_offset_weight=translation_offset_weight,
            rotation_offset_weight=rotation_offset_weight,
            plane_scale=plane_scale,
            plane_scale_offset=plane_scale_offset,
            translation=translation,
            orientation=orientation,
            position=position,
            times=times,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            index=index,
            **kwargs)
        self.box = box
        """Box node if the plane was created from a box."""

        self.ray_dependent = False
        """If the planes properties are considered ray dependent. If true, ray_origins and ray_directions must be supplied for forward pass."""
        self.network_dtype = network_dtype

        # If Background, then we have no alpha.
        self.encoding_image = tcnn.Encoding(
            n_input_dims=2, encoding_config=EncodingConfig.parse(encoding_image_config).to_dict(),
            dtype=network_dtype
        )
        self.encoding_flow = tcnn.Encoding(
            n_input_dims=2, encoding_config=EncodingConfig.parse(encoding_flow_config).to_dict(),
            dtype=network_dtype
        )

        if independent_rgba_flow:
            self.encoding_flow_alpha = tcnn.Encoding(
                n_input_dims=2, encoding_config=EncodingConfig.parse(encoding_flow_config).to_dict(),
                dtype=network_dtype
            )
        else:
            self.encoding_flow_alpha = None

        self.encoding_alpha = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=EncodingConfig.parse(
                encoding_alpha_config).to_dict(),
            dtype=network_dtype)

        self.num_flow_control_points = num_flow_control_points + \
            2  # Adding 2 for the borders

        self.deprecated_flow = deprecated_flow
        self.coarse_to_fine_color = coarse_to_fine_color
        self.coarse_to_fine_alpha = coarse_to_fine_alpha

        self.register_buffer("flow_timestamps", default_control_point_times(
            num_flow_control_points, dtype=self.dtype))

        self.register_buffer("flow_weight", tensorify(
            flow_weight, dtype=self.dtype))

        self.network_image = tcnn.Network(
            n_input_dims=self.encoding_image.n_output_dims, n_output_dims=3, network_config=NetworkConfig.parse(network_image_config).to_dict())
        self.network_flow = tcnn.Network(n_input_dims=self.encoding_flow.n_output_dims,
                                         n_output_dims=2*(num_flow_control_points + 2), network_config=NetworkConfig.parse(network_flow_config).to_dict())
        self.network_alpha = tcnn.Network(n_input_dims=self.encoding_alpha.n_output_dims,
                                          n_output_dims=1, network_config=NetworkConfig.parse(network_alpha_config).to_dict())
        if independent_rgba_flow:
            self.network_flow_alpha = tcnn.Network(
                n_input_dims=self.encoding_flow_alpha.n_output_dims,
                n_output_dims=2 * (num_flow_control_points + 2),
                network_config=NetworkConfig.parse(network_flow_config).to_dict())
        else:
            self.network_flow_alpha = None

        if flow_reference_time is None:
            flow_reference_time = 0.
        self.register_buffer("flow_reference_time", tensorify(
            flow_reference_time, dtype=self.dtype))

        self.rgb_rescaling = rgb_rescaling
        self.alpha_rescaling = alpha_rescaling
        self.flow_rescaling = flow_rescaling
        self.independent_rgba_flow = independent_rgba_flow

        self.legacy_model_inputs = False

        self.init_initial_alpha(self.dtype, initial_alpha)
        self.init_initial_rgb(
            self.dtype,
            initial_rgb,
            initial_rgb_variance=initial_rgb_variance,
            mean_rgb_variance=mean_rgb_variance)

        if rgb_weight is None:
            rgb_weight = torch.tensor(0.1, dtype=self.dtype)
        else:
            rgb_weight = tensorify(rgb_weight, dtype=self.dtype)
        if alpha_weight is None:
            alpha_weight = torch.tensor(0.1, dtype=self.dtype)
        else:
            alpha_weight = tensorify(alpha_weight, dtype=self.dtype)
        self.register_buffer("rgb_weight", rgb_weight)
        self.register_buffer("alpha_weight", alpha_weight)

        if self.rgb_rescaling:
            if rgb_normalization is None:
                rgb_normalization = MeanStd(dim=0, mean=0, std=DEFAULT)
            self.rgb_normalization = rgb_normalization
            if not proxy_init:
                self.estimate_rgb_scaling(rgb_normalization)
        else:
            self.rgb_normalization = None

        if self.alpha_rescaling:
            if alpha_normalization is None:
                alpha_normalization = MeanStd(dim=0, mean=0, std=DEFAULT)
            self.alpha_normalization = alpha_normalization
            if not proxy_init:
                self.estimate_alpha_scaling(alpha_normalization)
        else:
            self.alpha_normalization = None

        if self.flow_rescaling:
            if flow_normalization is None:
                flow_normalization = MeanStd(dim=0, mean=0, std=DEFAULT)
            self.flow_normalization = flow_normalization
            if self.independent_rgba_flow:
                flow_normalization_alpha = MeanStd(dim=0, mean=0, std=DEFAULT)
                self.flow_normalization_alpha = flow_normalization_alpha
            if not proxy_init:
                self.estimate_flow_scaling(self.flow_normalization)
                if self.independent_rgba_flow:
                    self.estimate_flow_scaling_alpha(
                        self.flow_normalization_alpha)
        else:
            self.flow_normalization = None
            self.flow_normalization_alpha = None
        self.alpha_rigid_model = alpha_rigid_model
        if alpha_rigid_model:
            self.alpha_rigid_parameters = torch.nn.Parameter(
                torch.zeros((len(self._times), 2), dtype=self.dtype))
            """Rigid shift offsets for the alpha plane. Shape: (T, 2)"""
        else:
            self.alpha_rigid_parameters = None

    def after_checkpoint_loaded(self, **kwargs):
        super().after_checkpoint_loaded(**kwargs)
        if self.rgb_rescaling and self.rgb_normalization is not None:
            self.rgb_normalization.fitted = True
        if self.alpha_rescaling and self.alpha_normalization is not None:
            self.alpha_normalization.fitted = True
        if self.flow_rescaling and self.flow_normalization is not None:
            self.flow_normalization.fitted = True

    def init_initial_alpha(self,
                           dtype: torch.dtype,
                           initial_alpha: Optional[torch.Tensor] = None,
                           ):
        if initial_alpha is None:
            initial_alpha = torch.tensor([[0.9]], dtype=dtype)
        self.register_buffer("initial_alpha", initial_alpha)

    def init_initial_rgb(self,
                         dtype: torch.dtype,
                         initial_rgb: Optional[torch.Tensor] = None,
                         initial_rgb_variance: Optional[torch.Tensor] = None,
                         mean_rgb_variance: Optional[torch.Tensor] = None,
                         ):
        if initial_rgb is None:
            initial_rgb = torch.zeros([1, 3], dtype=dtype)
        if initial_rgb_variance is None:
            initial_rgb_variance = torch.ones([1, 3], dtype=dtype)
        if mean_rgb_variance is None:
            mean_rgb_variance = torch.ones([1, 3], dtype=dtype)
        self.register_buffer("initial_rgb", initial_rgb)
        self.register_buffer("initial_rgb_variance", initial_rgb_variance)
        self.register_buffer("mean_rgb_variance", mean_rgb_variance)

    def get_initial_alpha(self, uv: torch.Tensor) -> torch.Tensor:
        return self.initial_alpha

    def get_initial_rgb(self, uv: torch.Tensor) -> torch.Tensor:
        return self.initial_rgb

    def estimate_rgb_scaling(self, normalization: FittableTransform):
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
            rgb = self.get_rgb(grid.reshape(
                H * W, 2), torch.tensor(0, device=device, dtype=self._translation.dtype))
            rgb_norm = normalization.fit_transform(rgb)
            if device != old_device:
                self.to(old_device)

    def estimate_alpha_scaling(self, normalization: FittableTransform):
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
            alpha = self.get_alpha(grid.reshape(
                H * W, 2), torch.tensor(0, device=device, dtype=self._translation.dtype))
            alpha_norm = normalization.fit_transform(alpha)
            if device != old_device:
                self.to(old_device)

    def estimate_flow_scaling(self, normalization: FittableTransform):
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
            if self.deprecated_flow:
                flow_field = self.get_legacy_flow(grid.reshape(
                    H * W, 2).unsqueeze(1), torch.tensor([self.flow_reference_time], device=device, dtype=self._translation.dtype), torch.tensor(0, device=device, dtype=self._translation.dtype))
                flow_field_norm = normalization.fit_transform(flow_field)
            else:
                flow_field = self._query_flow(
                    grid.reshape(H * W, 2).unsqueeze(1),
                    torch.tensor(0, device=device, dtype=self._translation.dtype))
                _ = normalization.fit_transform(flow_field.reshape(
                    H * W, self.num_flow_control_points, 2))
            if device != old_device:
                self.to(old_device)

    def estimate_flow_scaling_alpha(self, normalization: FittableTransform):
        """Estimates the flow scaling for the alpha plane."""
        if not self.independent_rgba_flow:
            raise ValueError(
                "Cannot estimate flow scaling for alpha plane if independent_rgba_flow is False.")
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
            flow_field = self._query_flow_alpha(
                grid.reshape(H * W, 2).unsqueeze(1),
                torch.tensor(0, device=device, dtype=self._translation.dtype))
            _ = normalization.fit_transform(flow_field.reshape(
                H * W, self.num_flow_control_points, 2))
            if device != old_device:
                self.to(old_device)

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

        B, T, _ = uv.shape
        # In local coordinates the plane center is at (0, 0, 0)
        if self.flow_weight != 0:
            if self.deprecated_flow:
                flow = self._compute_flow(uv, t, sin_epoch,
                                          right_idx_flow=right_idx_flow,
                                          rel_frac_flow=rel_frac_flow
                                          )
            else:
                flow = self.get_flow(uv,
                                     t,
                                     sin_epoch,
                                     is_inside=is_inside,
                                     right_idx_flow=right_idx_flow,
                                     rel_frac_flow=rel_frac_flow
                                     )
        else:
            flow = torch.zeros_like(uv)

        query_points = uv + flow  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        # And the flow as adjusted for the time already
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)

        if is_inside is not None:
            query_points = query_points[is_inside.reshape(B * T)]

        if query_points.numel() != 0:
            # Get the RGB values
            network_rgb = self.get_rgb(query_points, sin_epoch)
            rgb = self.get_initial_rgb(query_points) + \
                self.rgb_weight * network_rgb
            rgb = rgb.clamp(0, 1)

            if self.render_texture_map:
                rgb = self.get_rendered_texture_map(query_points, rgb)

            # Get the alpha values
            network_alpha = self.get_alpha(query_points, sin_epoch)
            initial_alpha = self.get_initial_alpha(
                query_points)

            alpha = torch.sigmoid(
                (-torch.log(1/initial_alpha - 1) + self.alpha_weight * network_alpha))
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

        # is_alpha_not_finite = ~torch.isfinite(alpha)
        # is_rgb_not_finite = ~torch.isfinite(rgb)
        # if is_alpha_not_finite.any() or (is_rgb_not_finite).any():
        #     self.logger.warning(
        #         f"Alpha or RGB not finite. Alpha: {is_alpha_not_finite.sum()}, RGB: {is_rgb_not_finite.sum()}")
        #     breakpoint()

        if context is not None:
            # Store alpha and flow for regularization if needed
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

    def forward(self,
                uv: torch.Tensor,
                ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                t: torch.Tensor,
                sin_epoch: torch.Tensor,
                uv_in_plane_space: bool = False,
                global_position: Optional[torch.Tensor] = None,
                plane_scale: Optional[torch.Tensor] = None,
                plane_scale_offset: Optional[torch.Tensor] = None,
                right_idx_flow: Optional[torch.Tensor] = None,
                rel_frac_flow: Optional[torch.Tensor] = None,
                next_sin_epoch: Optional[torch.Tensor] = None,
                batch_idx: Optional[int] = None,
                max_batch_idx: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None,
                is_inside: Optional[torch.Tensor] = None,
                **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the plane.

        Parameters
        ----------
        uv : torch.Tensor
            The intersection points / query points of the rays with the plane.
            In global coordinates unless uv_in_plane_space is True.
            Shape: (B, T, 3)
        ray_origins : torch.Tensor
            The ray origins of the rays. In global coordinates unless uv_in_plane_space is True.
            Shape: (B, T, 3)
        ray_directions : torch.Tensor
            The ray directions of the rays. In global coordinates unless uv_in_plane_space is True.
            Shape: (B, T, 3)
        t : torch.Tensor
            The time of the intersection.
            Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch.
            Tends towards 1 at the end of the epoch and towards -1 at the beginning of the epoch.

        uv_in_plane_space : bool
            If the uv coordinates are in plane space. If True, the uv coordinates are assumend in the range [0, 1].

        global_position : Optional[torch.Tensor]
            The global position of the plane.
            If None, the global position of the plane is computed from the position and orientation of the plane.
            Can be supplied to avoid recomputing the global position. Shape: (T, 4, 4)

        plane_scale : Optional[torch.Tensor]
            The scale of the plane. If None, the scale in the object is used.
            Can be supplied to reduce overhead. Shape: (2, ) (x, y)

        plane_scale_offset : Optional[torch.Tensor]
            The offset of the plane scale. If None, the offset in the object is used.
            Can be supplied to reduce overhead. Shape: (2, ) (x, y)

        is_inside : Optional[torch.Tensor]
            If the points hitting the plane actually within its bounds. Shape: (B, T)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. The RGB values of the plane. Shape: (B, T, 3)
            2. The alpha values of the plane. Shape: (B, T, 1)
        """
        B, T = uv.shape[:2]

        if uv.numel() == 0:
            return torch.zeros(B, T, 3, dtype=self.dtype, device=uv.device), torch.zeros(B, T, 1, dtype=self.dtype, device=uv.device)

        if not uv_in_plane_space:
            if global_position is None:
                global_position = self.get_global_position(t=t)

            if plane_scale is None:
                plane_scale = self.get_plane_scale()

            if plane_scale_offset is None:
                plane_scale_offset = self.get_plane_scale_offset()

            # Convert the global intersection points to local coordinates
            # (B, T, 2) (x, y) Ignore z and w. x, y should be in local coordinate system now
            uv_local = global_to_local(
                global_position, uv, v_include_time=True)[..., :2]

            # Convert local to plane space
            uv_plane = local_to_plane_coordinates(
                uv_local, plane_scale, plane_scale_offset)  # Go from local to plane coordinate system

            if self.ray_dependent:
                # Convert ray directions to plane space
                ray_global_target = ray_origins + ray_directions
                ray_local_target = global_to_local(
                    global_position, ray_global_target, v_include_time=True)[..., :3]
                ray_local_origin = global_to_local(
                    global_position, ray_origins, v_include_time=True)[..., :3]
                ray_local_directions = ray_local_target - ray_local_origin
                ray_directions = ray_local_directions
                ray_origins = ray_local_origin
            else:
                ray_directions = None
                ray_origins = None
        else:
            uv_plane = uv[..., :2]

        # Plane_intersection_points are in range [0, 1] for x, y, we need to convert them to [-0.5, 0.5]
        uv_network = uv_plane - 0.5

        return self.compute_rgb_alpha(uv=uv_network,
                                      t=t,
                                      sin_epoch=sin_epoch,
                                      right_idx_flow=right_idx_flow,
                                      rel_frac_flow=rel_frac_flow,
                                      ray_directions=ray_directions,
                                      ray_origins=ray_origins,
                                      context=context,
                                      is_inside=is_inside,
                                      )

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
        else:
            flow = torch.zeros_like(uv_network)

        query_points = uv_network + flow  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)
        # query_points = shadow_zeros(query_points)

        if query_color:
            # Get the RGB values
            network_rgb = self.get_rgb(query_points, sin_epoch)
            rgb = self.get_initial_rgb(query_points) + \
                self.rgb_weight * network_rgb
            rgb = rgb.clamp(0, 1)
            # Reshape to original shape
            rgb = rgb.reshape(B, T, 3)
        else:
            rgb = torch.zeros(B, T, 3, dtype=self.dtype,
                              device=uv_plane.device)

        if query_alpha:
            # Get the alpha values
            network_alpha = self.get_alpha(query_points, sin_epoch)
            initial_alpha = self.get_initial_alpha(
                query_points)

            alpha = torch.sigmoid(
                (-torch.log(1/initial_alpha - 1) + self.alpha_weight * network_alpha))
            alpha = alpha.reshape(B, T, 1)
        else:
            alpha = torch.zeros(B, T, 1, dtype=self.dtype,
                                device=uv_plane.device)

        return rgb, alpha, flow

    def get_rgb(self, uv: torch.Tensor, sin_epoch: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the RGB values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The RGB values for the given uv coordinates.
            Shape: (B, 3)
        """
        query_points = uv
        if not self.legacy_model_inputs:
            query_points = query_points + 0.5
        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            enc = self.encoding_image(query_points)
            if self.coarse_to_fine_color:
                enc = utils.mask(enc, sin_epoch)
            network_rgb = self.network_image(enc)

        network_rgb = network_rgb.to(dtype=self.dtype)
        if self.rgb_rescaling and self.rgb_normalization.fitted:
            network_rgb = self.rgb_normalization(network_rgb)
        return network_rgb

    def get_alpha(self,
                  uv: torch.Tensor,
                  sin_epoch: torch.Tensor,
                  **kwargs
                  ) -> torch.Tensor:
        """Get the alpha values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The alpha values for the given uv coordinates.
            Shape: (B, 1)
        """
        query_points = uv
        if not self.legacy_model_inputs:
            query_points = query_points + 0.5
        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            enc = self.encoding_alpha(query_points)
            if self.coarse_to_fine_alpha:
                enc = utils.mask(enc, sin_epoch)
            network_alpha = self.network_alpha(enc)
        network_alpha = network_alpha.to(dtype=self.dtype)

        if self.alpha_rescaling and self.alpha_normalization.fitted:
            network_alpha = self.alpha_normalization(network_alpha)
        return network_alpha

    def get_legacy_flow(self, uv: torch.Tensor, t: torch.Tensor, sin_epoch: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the flow values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point and time
            Shape: (B, T, 2) x, y should be in range [-0.5, 0.5]
            T must contain the flow reference time.

        t : torch.Tensor
            The times of the points. Shape: (T, )

        Returns
        -------
        torch.Tensor
            The flow values for the given uv coordinates.
            And flow control points.
            Shape: (B, FT, 2)
        """
        uv_reference_frame = uv[:, t == self.flow_reference_time].squeeze(1)
        if not self.legacy_model_inputs:
            uv_reference_frame = uv_reference_frame + 0.5

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            flow = self.network_flow(utils.mask(
                self.encoding_flow(uv_reference_frame), sin_epoch))  # (B, 2)
        flow = flow.to(dtype=self.dtype)

        flow = flow.reshape(-1, self.num_flow_control_points, 2)  # (B, FT, 2)
        if self.flow_rescaling and self.flow_normalization.fitted:
            flow = self.flow_normalization(flow)
        return flow

    def _query_flow(self,
                    uv: torch.Tensor,
                    sin_epoch: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """Query the flow values for the given uv coordinates.

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
            Flow control points for Time Interpolation. Shape: (B, T, FT, 2)
        """
        input_coords = uv  # (B, T, 2)
        input_coords = input_coords + 0.5
        B, T, _ = input_coords.shape
        FT = self.num_flow_control_points
        input_coords = input_coords.reshape(B * T, 2)

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            flow_points = self.network_flow(utils.mask(
                self.encoding_flow(input_coords), sin_epoch))  # (B * T, 2)

        flow_points = flow_points.to(dtype=self.dtype).reshape(
            B * T, FT, 2)  # (B * T, FT, 2)
        if self.flow_rescaling and self.flow_normalization.fitted:
            flow_points = self.flow_normalization(flow_points)

        flow_points = flow_points.reshape(B, T, FT, 2)  # (B, T, FT, 2)
        return flow_points

    def get_flow(self,
                 uv: torch.Tensor,
                 t: torch.Tensor,
                 sin_epoch: torch.Tensor,
                 right_idx_flow: Optional[torch.Tensor] = None,
                 rel_frac_flow: Optional[torch.Tensor] = None,
                 context: Optional[Dict[str, Any]] = None,
                 is_inside: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """Computes the flow for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of a point within the image plane.
            Shape: (B, T, 2) (x, y) in range [-0.5, 0.5]
        t : torch.Tensor
            The times of the points. Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch, progress marker.
        right_idx_flow : Optional[torch.Tensor], optional
            The right index for the interpolation, by default None
            As flow control points must not be the same as the rigid control points.
        rel_frac_flow : Optional[torch.Tensor], optional
            The relative fraction for the interpolation, by default None

        Returns
        -------
        torch.Tensor
            The flow offset for the given uv coordinates. Shape: (B, T, 2)
        """
        B, T = uv.shape[:2]
        FT = self.num_flow_control_points
        BT = B * T
        # Times for the flow are equidistant
        interpolate_times = t[None, :, None].repeat(B, 1, 1).reshape(BT, 1)
        if is_inside is not None:
            uv = uv[is_inside].unsqueeze(1)
            BT = uv.shape[0]
            interpolate_times = interpolate_times[is_inside.reshape(
                B * T)].unsqueeze(1)

        if uv.numel() != 0:
            flow_points = self._query_flow(uv, sin_epoch)  # (B, T, FT, 2)
            flow_points = flow_points.reshape(BT, FT, 2)

            # (B * T, 1)
            flow = self.flow_weight * \
                interpolate_vector(flow_points,
                                   self.flow_timestamps.unsqueeze(
                                       0).repeat(flow_points.shape[0], 1),
                                   interpolate_times,
                                   equidistant_times=True,
                                   method="cubic",
                                   right_idx=right_idx_flow,
                                   rel_frac=rel_frac_flow
                                   )
        else:
            flow = torch.zeros(BT, 2, dtype=self.dtype, device=uv.device)

        flow = flow.reshape(BT, 2)
        if context is not None:
            idx = self.get_index()
            if context.get("store_object_flow_control_points", False):
                if "object_flow_control_points" not in context:
                    context["object_flow_control_points"] = dict()
                context["object_flow_control_points"][idx] = flow_points
            if context.get("store_object_flow_lap", False):
                if "object_flow_lap" not in context:
                    context["object_flow_lap"] = dict()
                # Compute Discrete Laplacian along FT
                laplacian = torch.tensor(
                    [1, -2, 1], device=flow.device, dtype=flow.dtype)[None, None]  # (1, 1, 3)
                # Permute flow_points to (B * T, FT, 2) for convolution
                vs = flow_points.permute(0, 2, 1)  # (B * T, 2, FT)
                # Flatten the first 3 dimensions to apply the convolution
                # (B * T * 2, 1, FT)
                vs = vs.reshape(BT * 2, FT).unsqueeze(1)

                # (B * T * 2, 1, FT - 2)
                lap = F.conv1d(vs, laplacian, padding=0, bias=None)
                lap = lap.abs().sum(dim=-1).reshape(BT, 2)  # (BT, 2)
                lap = lap.mean(dim=-1)  # (BT)

                if is_inside is not None:
                    ret = torch.zeros(B, T, dtype=lap.dtype, device=lap.device)
                    ret[...] = torch.nan
                    ret[is_inside] = lap
                    lap = ret

                context["object_flow_lap"][idx] = lap

        if is_inside is None:
            return flow
        else:
            ret = torch.zeros(B, T, 2, dtype=flow.dtype, device=flow.device)
            ret[is_inside] = flow
            return ret

    def _query_flow_alpha(self,
                          uv: torch.Tensor,
                          sin_epoch: torch.Tensor,
                          **kwargs) -> torch.Tensor:
        """Query the dedicated alpha flow values for the given uv coordinates if independent_rgba_flow is True.

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
            Flow control points for Time Interpolation. Shape: (B, T, FT, 2)
        """
        input_coords = uv  # (B, T, 2)
        input_coords = input_coords + 0.5
        B, T, _ = input_coords.shape
        FT = self.num_flow_control_points
        input_coords = input_coords.reshape(B * T, 2)

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            flow_points = self.network_flow_alpha(utils.mask(
                self.encoding_flow_alpha(input_coords), sin_epoch))  # (B * T, 2)

        flow_points = flow_points.to(dtype=self.dtype).reshape(
            B * T, FT, 2)  # (B * T, FT, 2)
        if self.flow_rescaling and self.flow_normalization_alpha.fitted:
            flow_points = self.flow_normalization_alpha(flow_points)

        flow_points = flow_points.reshape(B, T, FT, 2)  # (B, T, FT, 2)
        return flow_points

    def get_rigid_alpha(self, uv: torch.Tensor, times: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the rigid alpha values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point.
            Shape: (B, T, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The rigid time translations for the given uv coordinates.
        """
        shifts = interpolate_vector(self.alpha_rigid_parameters.unsqueeze(0),
                                    self._times,
                                    times.unsqueeze(0),
                                    equidistant_times=True,
                                    method="cubic",
                                    right_idx=None,
                                    rel_frac=None)
        return shifts.expand_as(uv)  # (B, T, 2)

    def get_flow_alpha(self,
                       uv: torch.Tensor,
                       t: torch.Tensor,
                       sin_epoch: torch.Tensor,
                       right_idx_flow: Optional[torch.Tensor] = None,
                       rel_frac_flow: Optional[torch.Tensor] = None,
                       context: Optional[Dict[str, Any]] = None,
                       is_inside: Optional[torch.Tensor] = None,
                       **kwargs) -> torch.Tensor:
        """Computes the alpha flow for the given uv coordinates if independent_rgba_flow is True.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of a point within the image plane.
            Shape: (B, T, 2) (x, y) in range [-0.5, 0.5]
        t : torch.Tensor
            The times of the points. Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch, progress marker.
        right_idx_flow : Optional[torch.Tensor], optional
            The right index for the interpolation, by default None
            As flow control points must not be the same as the rigid control points.
        rel_frac_flow : Optional[torch.Tensor], optional
            The relative fraction for the interpolation, by default None

        Returns
        -------
        torch.Tensor
            The flow offset for the given uv coordinates. Shape: (B, T, 2)
        """
        B, T = uv.shape[:2]
        FT = self.num_flow_control_points
        BT = B * T
        # Times for the flow are equidistant
        interpolate_times = t[None, :, None].repeat(B, 1, 1).reshape(BT, 1)
        if is_inside is not None:
            uv = uv[is_inside].unsqueeze(1)
            BT = uv.shape[0]
            interpolate_times = interpolate_times[is_inside.reshape(
                B * T)].unsqueeze(1)

        if uv.numel() != 0:
            flow_points = self._query_flow_alpha(
                uv, sin_epoch)  # (B, T, FT, 2)
            flow_points = flow_points.reshape(BT, FT, 2)

            # (B * T, 1)
            flow = self.flow_weight * \
                interpolate_vector(flow_points,
                                   self.flow_timestamps.unsqueeze(
                                       0).repeat(flow_points.shape[0], 1),
                                   interpolate_times,
                                   equidistant_times=True,
                                   method="cubic",
                                   right_idx=right_idx_flow,
                                   rel_frac=rel_frac_flow
                                   )
        else:
            flow = torch.zeros(BT, 2, dtype=self.dtype, device=uv.device)

        flow = flow.reshape(BT, 2)

        if is_inside is None:
            return flow
        else:
            ret = torch.zeros(B, T, 2, dtype=flow.dtype, device=flow.device)
            ret[is_inside] = flow
            return ret

    def _compute_flow(self,
                      uv: torch.Tensor,
                      t: torch.Tensor,
                      sin_epoch: torch.Tensor,
                      right_idx_flow: Optional[torch.Tensor] = None,
                      rel_frac_flow: Optional[torch.Tensor] = None,
                      context: Optional[Dict[str, Any]] = None,
                      **kwargs) -> torch.Tensor:
        """Computes the flow for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of a point within the image plane.
            Shape: (B, T, 2) (x, y) in range [-0.5, 0.5]
            T must contain the flow reference time.
        t : torch.Tensor
            The times of the points. Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch, progress marker.
        right_idx_flow : Optional[torch.Tensor], optional
            The right index for the interpolation, by default None
            As flow control points must not be the same as the rigid control points.
        rel_frac_flow : Optional[torch.Tensor], optional
            The relative fraction for the interpolation, by default None

        Returns
        -------
        torch.Tensor
            The flow offset for the given uv coordinates.
        """
        # We query the flow network with the first time point. As the flow network outputs multiple points which can be interpolated.

        flow_points = self.get_legacy_flow(uv, t, sin_epoch)  # (B, FT, 2)
        B, FT, _ = flow_points.shape
        # Times for the flow are equidistant
        flow = self.flow_weight * \
            interpolate_vector(flow_points,
                               self.flow_timestamps.unsqueeze(
                                   0).repeat(flow_points.shape[0], 1),
                               t.unsqueeze(0).repeat(flow_points.shape[0], 1),
                               equidistant_times=True,
                               method="cubic",
                               right_idx=right_idx_flow,
                               rel_frac=rel_frac_flow)
        if context is not None:
            idx = self.get_index()
            if context.get("store_object_flow_control_points", False):
                if "object_flow_control_points" not in context:
                    context["object_flow_control_points"] = dict()
                context["object_flow_control_points"][idx] = flow_points
        return flow

    def _get_estimated_alpha(self, uv: torch.Tensor) -> torch.Tensor:
        uv, shape = flatten_batch_dims(uv, -2)
        network_alpha = self.get_alpha(uv, torch.tensor(
            1.0, device=uv.device, dtype=uv.dtype))
        alpha = torch.sigmoid(
            (-torch.log(1/self.get_initial_alpha(uv) - 1) + self.alpha_weight * network_alpha))
        return unflatten_batch_dims(alpha, shape).detach().cpu()

    def _get_estimated_rgb(self, uv: torch.Tensor) -> torch.Tensor:
        uv, shape = flatten_batch_dims(uv, -2)
        rgb = self.get_rgb(uv, torch.tensor(
            1.0, device=uv.device, dtype=uv.dtype))
        rgb = self.get_initial_rgb(uv) + self.rgb_weight * rgb
        rgb = rgb.clamp(0, 1)
        return unflatten_batch_dims(rgb, shape).detach().cpu()

    # endregion

    # region Plotting

    def _plot_plane(self,
                    ax: Axes,
                    t: Optional[torch.Tensor] = None,
                    facecolor: str = 'white',
                    edgecolor: str = 'black',
                    alpha: float = 0.3,
                    line_width: float = 1.0,
                    **kwargs):
        """Plots the corners of the plane on the given axis.

        Parameters
        ----------
        ax : Axes
            The axis to plot on.
        corners : torch.Tensor
            The corners of the plane to plot.
        """
        from matplotlib.colors import to_rgba, get_named_colors_mapping
        facec = facecolor
        plot_plane_alpha = kwargs.get('plot_plane_alpha', False)
        plot_plane_image = kwargs.get('plot_plane_image', False)

        if plot_plane_alpha or plot_plane_image:
            # Remove base facec
            facec = np.array([0, 0, 0, 0])
            alpha = 0.

        super()._plot_plane(ax=ax, t=t, facecolor=facec, edgecolor=edgecolor,
                            alpha=alpha, line_width=line_width, **kwargs)

        if kwargs.get('plot_box', False):
            if self.box is not None:
                self.box.plot_object(ax=ax, t=t, box_color=edgecolor, **kwargs)

        if plot_plane_alpha or plot_plane_image:
            sub_H, sub_W = 200, 200

            subsample = kwargs.get('plot_plane_image_subsample', 10)

            xx, yy = torch.meshgrid(torch.arange(
                0, 1, 1 / sub_W * subsample, device=self._translation.device), torch.arange(0, 1, 1 / sub_H * subsample, device=self._translation.device), indexing="xy")
            # Add half a pixel to get the center of the pixel
            xx = xx + 1 / (2 * sub_W * subsample)
            yy = yy + 1 / (2 * sub_H * subsample)

            xy = torch.stack([xx, yy], dim=-1)
            # XYZ to local coordinates
            xyz = self.plane_coordinates_to_local(xy)

            # Add w=1
            xyzw = torch.cat(
                [xyz, torch.ones_like(xyz[..., :1])], dim=-1)

            xyz_glob = self.local_to_global(
                xyzw, t=t)[..., 0, :]  # Ignore time

            faces = torch.zeros(4, math.ceil(
                sub_H / subsample), math.ceil(sub_W / subsample), dtype=torch.float32, device=self._translation.device)  # RGBA

            if plot_plane_alpha or plot_plane_image:
                try:
                    old_device = self._translation.device
                    device = torch.device("cuda")
                    # Move model to cuda if not already
                    if device != old_device:
                        self.to(device)
                    xy = xy.to(device)

                    uv = xy - 0.5
                    if plot_plane_alpha and not plot_plane_image:
                        faces[3] = self._get_estimated_alpha(
                            uv)[..., 0]  # H x W x 1
                        faces[:3] = 1.0
                    else:
                        faces[3] = 1.0

                    if plot_plane_image:

                        if plot_plane_alpha:
                            faces[:3] = self._get_estimated_rgb(
                                uv).permute(2, 0, 1)
                            faces[3] = self._get_estimated_alpha(
                                uv)[..., 0]  # H x W x 1
                        else:
                            faces[:3] = self._get_estimated_rgb(
                                uv).permute(2, 0, 1)
                finally:
                    self.to(old_device)
            else:
                fc = facecolor
                if isinstance(fc, str):
                    fc = torch.from_numpy(np.array(to_rgba(fc)))
                faces[:3] = fc[:3].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                      *faces.shape[-2:])
            # Plot the surface
            xyz_glob = numpyify(xyz_glob)
            ax.plot_surface(xyz_glob[..., 0], xyz_glob[..., 1], xyz_glob[..., 2], rstride=1,
                            cstride=1, facecolors=numpyify(faces.permute(1, 2, 0)), shade=False)
    # endregion
