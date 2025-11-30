import math
from matplotlib.axes import Axes
import numpy as np
import torch
from nag.config.encoding_config import EncodingConfig
from nag.config.network_config import NetworkConfig
from nag.model.background_plane_scene_node_3d import BackgroundPlaneSceneNode3D
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING, Tuple, Union
from tools.transforms.fittable_transform import FittableTransform

from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import compose_translation_orientation, global_to_local
from tools.transforms.geometric.mappings import unitquat_to_rotvec
from tools.logger.logging import logger
try:
    import tinycudann as tcnn
except (ModuleNotFoundError, OSError) as err:
    from tools.util.mock_import import MockImport
    if not TYPE_CHECKING:
        logger.warning(f"Could not import tinycudann: {err}")
        tcnn = MockImport(mocked_property="tcnn")
from tools.util.typing import VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.to_tensor import tensorify
from tools.transforms.mean_std import MeanStd
from tools.util.typing import DEFAULT, _DEFAULT
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims
from nag.utils import utils
from nag.transforms.transforms_timed_3d import interpolate_vector
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates
from tools.transforms.to_numpy import numpyify
from tools.transforms.geometric.quaternion import quat_average, quat_subtraction
from nag.model.texture_mappable_scene_node_3d import TextureMappableSceneNode3D


def default_control_point_times(
        num_control_points: int,
        dtype: torch.dtype) -> torch.Tensor:
    if num_control_points > 1:
        t_step = 1. / (num_control_points - 1)
        return torch.linspace(0 - t_step, 1 + t_step, num_control_points + 2, dtype=dtype)
    else:
        return torch.tensor([-1, 0, 1], dtype=dtype)


class BackgroundImagePlaneSceneNode3D(BackgroundPlaneSceneNode3D, TextureMappableSceneNode3D):
    """"Background image plane class with a neural field for color and flow, without learnable position parameter."""

    encoding_image: tcnn.Encoding
    """The encoding for the image plane"""

    encoding_flow: tcnn.Encoding
    """The encoding for the flow of the plane."""

    flow_timestamps: torch.Tensor
    """The timestamps for the flow control points. Shape: (num_flow_control_points, ) in range [0, 1]."""

    flow_weight: torch.Tensor
    """The weight factor for the flow offset. Shape (1,) / float."""

    flow_reference_time: torch.Tensor
    """The reference time for the flow interpolation. If None, the time 0 is used."""

    flow_weight: torch.Tensor
    """The weight factor for the flow offset. Shape (1,) / float."""

    flow_reference_time: torch.Tensor
    """The reference time for the flow interpolation. If None, the time 0 is used."""

    network_image: tcnn.Network
    """The tcnn network for predicting the image color plane."""

    network_flow: tcnn.Network
    """The tcnn network for predicting the flow on the plane."""

    network_dtype: torch.dtype
    """The dtype for the network."""

    # region Initialization and Basic Getters

    def __init__(
        self,
            num_flow_control_points: int,
            encoding_image_config: Union[EncodingConfig, str] = "small",
            encoding_flow_config: Union[EncodingConfig, str] = "small",
            network_image_config: Union[NetworkConfig, str] = "small",
            network_flow_config: Union[NetworkConfig, str] = "small",
            rgb_rescaling: bool = True,
            rgb_normalization: Optional[FittableTransform] = None,
            flow_weight: float = 0.1,
            flow_input_dims: int = 2,
            flow_reference_time: Optional[VEC_TYPE] = None,
            flow_rescaling: bool = True,
            flow_normalization: Optional[FittableTransform] = None,
            flow_normalization_init: bool = True,
            flow_has_control_points: bool = True,
            coarse_to_fine_color: bool = False,
            plane_scale: Optional[VEC_TYPE] = None,
            plane_scale_offset: Optional[VEC_TYPE] = None,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            initial_rgb: Optional[torch.Tensor] = None,
            initial_rgb_variance: Optional[torch.Tensor] = None,
            mean_rgb_variance: Optional[torch.Tensor] = None,
            rgb_weight: Optional[torch.Tensor] = None,
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
            align_corners: bool = True,
            **kwargs
    ):
        super().__init__(
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
            declare_background_color=False,
            **kwargs)
        self.ray_dependent = False
        self.align_corners = align_corners
        self.network_dtype = network_dtype

        # If Background, then we have no alpha.
        self.encoding_image = tcnn.Encoding(
            n_input_dims=2, encoding_config=EncodingConfig.parse(encoding_image_config).to_dict(),
            dtype=network_dtype
        )
        self.encoding_flow = tcnn.Encoding(
            n_input_dims=flow_input_dims, encoding_config=EncodingConfig.parse(
                encoding_flow_config).to_dict(),
            dtype=network_dtype
        )
        self.coarse_to_fine_color = coarse_to_fine_color

        self.deprecated_flow = deprecated_flow

        flow_output_dims = 0
        if flow_has_control_points:
            self.register_buffer("flow_timestamps", default_control_point_times(
                num_flow_control_points, dtype=self.dtype))
            self.num_flow_control_points = num_flow_control_points + 2
            flow_output_dims = 2*(num_flow_control_points + 2)
        else:
            flow_output_dims = 2

        self.register_buffer("flow_weight", tensorify(
            flow_weight, dtype=self.dtype))

        self.network_image = tcnn.Network(
            n_input_dims=self.encoding_image.n_output_dims, n_output_dims=3, network_config=NetworkConfig.parse(network_image_config).to_dict())
        self.network_flow = tcnn.Network(n_input_dims=self.encoding_flow.n_output_dims,
                                         n_output_dims=flow_output_dims, network_config=NetworkConfig.parse(network_flow_config).to_dict())

        if flow_reference_time is None:
            flow_reference_time = 0.

        self.register_buffer("flow_reference_time", tensorify(
            flow_reference_time, dtype=self.dtype))

        self.rgb_rescaling = rgb_rescaling
        self.flow_rescaling = flow_rescaling

        self.legacy_model_inputs = False

        self.init_initial_rgb(
            self.dtype,
            initial_rgb,
            initial_rgb_variance=initial_rgb_variance,
            mean_rgb_variance=mean_rgb_variance)

        if rgb_weight is None:
            rgb_weight = torch.tensor(0.1, dtype=self.dtype)
        else:
            rgb_weight = tensorify(rgb_weight, dtype=self.dtype)

        self.register_buffer("rgb_weight", rgb_weight)

        if self.rgb_rescaling:
            if rgb_normalization is None:
                rgb_normalization = MeanStd(dim=0, mean=0, std=DEFAULT)
            self.rgb_normalization = rgb_normalization
            if not proxy_init:
                self.estimate_rgb_scaling(rgb_normalization)
        else:
            self.rgb_normalization = None

        if self.flow_rescaling:
            if flow_normalization is None:
                flow_normalization = MeanStd(dim=0, mean=0, std=DEFAULT)
            self.flow_normalization = flow_normalization
            if not proxy_init and flow_normalization_init:
                self.estimate_flow_scaling(flow_normalization)
        else:
            self.flow_normalization = None

    def after_checkpoint_loaded(self, **kwargs):
        super().after_checkpoint_loaded(**kwargs)
        if self.rgb_rescaling and self.rgb_normalization is not None:
            self.rgb_normalization.fitted = True
        if self.flow_rescaling and self.flow_normalization is not None:
            self.flow_normalization.fitted = True

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

    def get_initial_rgb(self, uv: torch.Tensor) -> torch.Tensor:
        """Gets the initial rgb for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            Uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The rgb values in shape (B, 3)
        """
        grid = (uv * 2)[None, None, ...]  # (1, 1, B, 2) to match gridsample
        out = torch.nn.functional.grid_sample(self.initial_rgb.unsqueeze(0),
                                              grid, mode="bilinear",
                                              padding_mode="border",
                                              align_corners=self.align_corners)[0].reshape(3, -1)
        # Swap channel dimension to the end
        out = out.permute(1, 0)
        return out

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
                _ = normalization.fit_transform(flow_field)
            else:
                flow_field = self._query_flow(
                    grid.reshape(H * W, 2).unsqueeze(1),
                    torch.tensor(0, device=device, dtype=self._translation.dtype))
                _ = normalization.fit_transform(flow_field.reshape(
                    H * W, self.num_flow_control_points, 2))
            if device != old_device:
                self.to(old_device)

    # region Forward

    def get_rgb(self, uv: torch.Tensor, sin_epoch: torch.Tensor) -> torch.Tensor:
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

    def get_legacy_flow(self, uv: torch.Tensor, t: torch.Tensor, sin_epoch: torch.Tensor) -> torch.Tensor:
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

    def _get_estimated_rgb(self, uv: torch.Tensor) -> torch.Tensor:
        uv, shape = flatten_batch_dims(uv, -2)
        rgb = self.get_rgb(uv, torch.tensor(
            1.0, device=uv.device, dtype=uv.dtype))
        rgb = self.get_initial_rgb(uv) + self.rgb_weight * rgb
        rgb = rgb.clamp(0, 1)
        return unflatten_batch_dims(rgb, shape).detach().cpu()

    def _compute_flow(self,
                      uv: torch.Tensor,
                      t: torch.Tensor,
                      sin_epoch: torch.Tensor,
                      right_idx_flow: Optional[torch.Tensor] = None,
                      rel_frac_flow: Optional[torch.Tensor] = None,
                      context: Optional[Dict[str, Any]] = None,
                      **kwargs
                      ) -> torch.Tensor:
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
        flow_points = self._query_flow(uv, sin_epoch)  # (B, T, FT, 2)
        B, T, FT, _ = flow_points.shape
        flow_points = flow_points.reshape(B * T, FT, 2)
        # Times for the flow are equidistant
        interpolate_times = t[None, :, None].repeat(
            B, 1, 1).reshape(B * T, 1)  # (B * T, 1)
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
        flow = flow.reshape(B, T, 2)
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
                vs = vs.reshape(B * T * 2, FT).unsqueeze(1)

                # (B * T * 2, 1, FT - 2)
                lap = F.conv1d(vs, laplacian, padding=0, bias=None)
                lap = lap.abs().sum(dim=-1).reshape(B, T, 2)  # (B, T, 2)
                lap = lap.mean(dim=-1)  # (B, T)
                context["object_flow_lap"][idx] = lap
        return flow

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

        # Get the RGB values
        network_rgb = self.get_rgb(query_points, sin_epoch)
        rgb = self.get_initial_rgb(query_points) + \
            self.rgb_weight * network_rgb
        rgb = rgb.clamp(0, 1)
        if self.render_texture_map:
            rgb = self.get_rendered_texture_map(query_points, rgb)

        rgb = rgb.reshape(B, T, 3)
        # Get the alpha values
        alpha = torch.ones(B, T, 1, dtype=self.dtype,
                           device=uv.device)

        # is_alpha_not_finite = ~torch.isfinite(alpha)
        # is_rgb_not_finite = ~torch.isfinite(rgb)
        # if is_alpha_not_finite.any() or (is_rgb_not_finite).any():
        #     self.logger.warning(
        #         f"Alpha or RGB not finite. Alpha: {is_alpha_not_finite.sum()}, RGB: {is_rgb_not_finite.sum()}")
        #     breakpoint()

        if context is not None:
            idx = self.get_index()
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
                **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the plane.

        Parameters
        ----------
        uv : torch.Tensor
            The intersection points / query points of the rays with the plane.
            In global coordinates.
            Shape: (B, T, 3)
        ray_origins : torch.Tensor
            The ray origins of the rays.
            Shape: (B, T, 3)
        ray_directions : torch.Tensor
            The ray directions of the rays.
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

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. The RGB values of the plane. Shape: (B, T, 3)
            2. The alpha values of the plane. Shape: (B, T, 1)
        """
        B, T = uv.shape[:2]

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
                                      ray_origins=ray_origins,
                                      ray_directions=ray_directions,
                                      context=context)

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

        alpha = torch.zeros(B, T, 1, dtype=self.dtype,
                            device=uv_plane.device)

        return rgb, alpha, flow

    # endregion

    # region For Camera Init

    @classmethod
    def for_camera(cls,
                   camera: TimedCameraSceneNode3D,
                   images: torch.Tensor,
                   masks: torch.Tensor,
                   depths: torch.Tensor,
                   times: torch.Tensor,
                   dataset: Any,
                   scene_cutoff_distance: float = 1.5,
                   dtype: torch.dtype = torch.float32,
                   relative_scale_margin: float = 0.1,
                   world: Optional[AbstractSceneNode] = None,
                   name: Optional[str] = None,
                   index: Optional[int] = None,
                   nag_model: Optional["Any"] = None,
                   color_mask_resolution: Optional[VEC_TYPE] = None,
                   color_mask_smoothing: bool = True,
                   num_flow_control_points: Optional[int] = None,
                   config: Any = None,
                   correct_lens_distortion: bool = False,
                   proxy_init: bool = False,
                   **kwargs):
        from nag.strategy.tilted_plane_initialization_strategy import reproject_alpha, reproject_color, reproject_color_new
        from nag.strategy.fixed_plane_position_strategy import FixedPlanePositionStrategy, compute_fixed_plane_scale
        from nag.strategy.plane_position_strategy import mask_to_camera_coordinates
        from tools.transforms.geometric.transforms3d import vector_angle, find_plane
        from tools.transforms.geometric.mappings import axis_angle_to_unitquat
        from tools.transforms.geometric.quaternion import quat_action
        if color_mask_resolution is None:
            color_mask_resolution = [200, 200]
        if not isinstance(color_mask_resolution, _DEFAULT):
            color_mask_resolution = tensorify(
                color_mask_resolution, dtype=dtype)
        if num_flow_control_points is None:
            num_flow_control_points = int(
                round(len(times) * config.plane_flow_control_points_ratio))
        with torch.no_grad():
            # Check, If camera rotation is close to 180 degrees, a plane model is not suitable
            # As the 0 th frame has no rotation, we can directly check the rotations of all frames
            if (torch.rad2deg(unitquat_to_rotvec(camera._orientation)) > (180 - 10)).any():
                logger.warning(
                    "Camera rotation is close to 180 degrees. A plane model is not suitable for this camera.")

            norm_vec_cand = quat_average(camera._orientation)
            # Get closests
            angle_diff = unitquat_to_rotvec(quat_subtraction(camera._orientation,
                                                             norm_vec_cand.expand(camera._orientation.shape[0], -1)))  # Check
            closests = torch.argmin(angle_diff.norm(dim=-1), dim=0)
            # Take the closests
            closests_t = camera._times[closests]
            center_ro, center_dir = camera.get_global_rays(uv=camera.get_intrinsics(
                t=closests_t)[:, :2, 2], t=closests_t, uv_includes_time=True)

            # position = center_ro + scene_cutoff_distance * center_dir

            border_coords_cam = mask_to_camera_coordinates(torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=dtype, device=times.device).unsqueeze(1).expand(-1, len(times), -1), torch.tensor(
                [1., 1.], dtype=dtype, device=times.device), camera._image_resolution).flip(-1)

            coord_ro, coord_rd = camera.get_global_rays(
                uv=border_coords_cam.swapaxes(0, 1), t=times, uv_includes_time=True)  # Shape B, T, 3

            dist = torch.tensor([0, 0, scene_cutoff_distance],
                                dtype=dtype, device=times.device)
            gdist = camera.local_to_global(
                dist, t=closests_t, v_includes_time=False)[..., :3]  # Shape 1, 1, 3
            cen = camera.get_global_position(t=closests_t)[..., :3, 3]
            gdir = gdist - cen  # Shape 1, 1, 3

            # Angle in the plane defined between the rays and the center ray
            angles = vector_angle(gdir.expand_as(
                coord_rd), coord_rd, mode="tan2")

            fac = gdir.norm(dim=-1) / torch.cos(angles)
            result_points = coord_ro + fac.unsqueeze(-1) * coord_rd
            plane_center, plane_normal = find_plane(result_points[:, 0])
            ###########################

            orientation = camera._orientation[closests].clone()

            global_plane_position = compose_translation_orientation(
                plane_center, orientation.unsqueeze(0)).expand(len(times), -1, -1)
            border_coords = torch.tensor([[0, 0], [0, 1], [1, 1], [
                                         1, 0]], dtype=dtype, device=times.device)  # bl, br, tr, tl (y, x)
            plane_scale = compute_fixed_plane_scale(border_coords=border_coords.unsqueeze(1).expand(-1, len(times), -1),
                                                    resolution=torch.tensor(
                                                        [1., 1.], dtype=dtype, device=times.device),
                                                    camera=camera, times=times,
                                                    global_plane_position=global_plane_position,
                                                    relative_plane_margin=relative_scale_margin)
            if color_mask_resolution == DEFAULT:
                # Take the image resolution and multiply it by the relative scale margin
                if config.use_dataset_color_reprojection:
                    res_x = camera._image_resolution[1]
                    res_y = camera._image_resolution[0]
                else:
                    res_x = min(camera._image_resolution[1], images.shape[3])
                    res_y = min(camera._image_resolution[0], images.shape[2])
                res = torch.tensor(
                    [res_y, res_x], dtype=dtype, device=times.device)
                color_mask_resolution = (
                    res * (1 + relative_scale_margin)).round().int()
                logger.info(
                    f"Color resolution on Background set to default. Using resolution {' x '.join([str(x.item()) for x in color_mask_resolution])} (H x W).")

            largest_mask_idx = 0

            if not proxy_init:
                background_mask = (masks.sum(dim=1, keepdim=True) == 0)
                _, projected_alpha_masks = reproject_alpha(
                    background_mask,
                    global_plane_positions=global_plane_position,
                    plane_scale=plane_scale,
                    resolution=color_mask_resolution,
                    times=times,
                    camera=camera,
                    largest_mask_idx=largest_mask_idx,
                    smooth=False,
                    correct_lens_distortion=correct_lens_distortion
                )
                if not config.use_dataset_color_reprojection:
                    projected_color_mask = reproject_color(
                        images=images,
                        masks=background_mask,
                        projected_masks=projected_alpha_masks,
                        global_plane_positions=global_plane_position,
                        plane_scale=plane_scale,
                        resolution=color_mask_resolution,
                        times=times,
                        camera=camera,
                        largest_mask_idx=largest_mask_idx,
                        fill_masked_with_closests_frame=True,
                        smooth=color_mask_smoothing,
                        temporal_consistency=False,
                        smooth_projected_masks=True,
                        correct_lens_distortion=correct_lens_distortion,
                    )
                else:
                    projected_color_mask = reproject_color_new(
                        images=images,
                        masks=background_mask,
                        projected_masks=projected_alpha_masks,
                        global_plane_positions=global_plane_position,
                        plane_scale=plane_scale,
                        resolution=color_mask_resolution,
                        times=times,
                        camera=camera,
                        largest_mask_idx=largest_mask_idx,
                        fill_masked_with_closests_frame=True,
                        smooth=color_mask_smoothing,
                        temporal_consistency=False,
                        smooth_projected_masks=True,
                        correct_lens_distortion=correct_lens_distortion,
                        dataset=dataset,
                        use_dataset=True,
                        align_corners=config.plane_align_corners,
                    )

            else:
                projected_color_mask = torch.zeros(
                    (3,) + tuple(color_mask_resolution), dtype=dtype, device=times.device)

            args = dict(kwargs)

            position_t = plane_center.expand(len(camera._times), -1)
            orientation_t = orientation.unsqueeze(
                0).expand(len(camera._times), -1)

            args.update(
                is_camera_attached=False,
                plane_scale=plane_scale,
                translation=position_t,
                times=camera._times,
                orientation=orientation_t,
                initial_rgb=projected_color_mask,
                flow_reference_time=largest_mask_idx,
                num_flow_control_points=num_flow_control_points,
                network_dtype=config.tinycudann_network_dtype,
                dtype=dtype,
                name=name,
                index=index)

            if nag_model is not None:
                from nag.model.nag_functional_model import NAGFunctionalModel
                if isinstance(nag_model, NAGFunctionalModel):
                    args = nag_model.patch_background_args(args)

            # Create the background plane
            plane = cls(
                **args
            )

            global_corners = plane.get_global_plane_corners(
                t=closests_t).detach()
            cam_image_corners, oks = camera.global_to_image_coordinates(
                global_corners, t=closests_t, v_includes_time=True, return_ok=True)
            cam_image_corners = cam_image_corners[:, :, :2].detach()

            world.add_scene_children(plane)

            return plane

    # endregion

    # region Plotting

    def _plot_plane(self,
                    ax: Axes,
                    t: Optional[torch.Tensor] = None,
                    edgecolor: str = 'black',
                    alpha: float = 1.,
                    line_width: float = 1.0,
                    facecolor: Optional[Any] = None,
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
        plot_plane_image = kwargs.get('plot_plane_image', False)

        facec = np.array([0, 0, 0, 0])

        super()._plot_plane(ax=ax, t=t, facecolor=facec, edgecolor=edgecolor,
                            alpha=0., line_width=line_width, **kwargs)

        if plot_plane_image:
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

            if plot_plane_image:
                try:
                    old_device = self._translation.device
                    device = torch.device("cuda")
                    # Move model to cuda if not already
                    if device != old_device:
                        self.to(device)
                    xy = xy.to(device)

                    uv = xy - 0.5
                    faces[3] = 1.0  # Alpha

                    if plot_plane_image:
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
